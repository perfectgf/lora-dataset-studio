"""Klein model resolution + auto-download preflight.

The lifted 'improve skin.json' workflow hardcodes the developer's own ComfyUI
filenames (diffusion_models/Flux2 klein/, flux2_vae.safetensors.safetensors,
klein/realistic.safetensors); a fresh install has a DIFFERENT layout. These tests
pin that (a) the helper resolves each loader node to what's actually on disk,
(b) a missing graph-critical model raises KleinModelsMissing, and (c) the route
turns that into a 409 that auto-starts the downloads (public now, gated model only
when an HF_TOKEN exists)."""
import io
import os
import struct

import pytest
from PIL import Image

# Smallest structurally-valid safetensors header (8-byte LE length + '{}'), so a
# fixture file reads as REAL (if tiny) weights. model_integrity now treats a bare
# stub as a broken file, and the readiness gate keys off it — the default here keeps
# every "asset present" fixture honest without writing multi-GB test data.
_VALID_ST = struct.pack('<Q', 2) + b'{}'


def _png(color=(0, 128, 255)):
    buf = io.BytesIO(); Image.new('RGB', (64, 64), color).save(buf, 'PNG')
    return buf.getvalue()


def _install(base, *relparts, data=_VALID_ST):
    p = base.joinpath(*relparts)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def _comfy(tmp_path, cfg, unet=True, vae=True, te=True, lora=False):
    """A ComfyUI tree with a configurable subset of the Klein assets present."""
    base = tmp_path / 'comfyui'
    (base / 'input').mkdir(parents=True)
    (base / 'output').mkdir(parents=True)
    (base / 'models').mkdir(parents=True)
    (base / 'main.py').write_text('# fake', encoding='utf-8')
    if unet:
        _install(base, 'models', 'unet', 'klein', 'flux-2-klein-9b-fp8.safetensors')
    if vae:
        _install(base, 'models', 'vae', 'flux2-vae.safetensors')
    if te:
        _install(base, 'models', 'text_encoders', 'qwen_3_8b_fp8mixed.safetensors')
    if lora:
        _install(base, 'models', 'loras', 'klein', 'Flux2-Klein-9B-consistency-V2.safetensors')
        _install(base, 'models', 'loras', 'klein', 'realistic.safetensors')
    cfg.save_config({'comfyui': {'base_dir': str(base)}})
    return base


# --- Resolvers -------------------------------------------------------------
def test_resolve_unet_prefixes_the_klein_subfolder(app, tmp_path):
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    with app.app_context():
        _comfy(tmp_path, cfg)
        # A UNETLoader lists files relative to models/unet -> the value MUST carry
        # the 'klein\' subfolder prefix, not the bare filename the picker sends.
        assert keh.resolve_klein_unet() == os.path.join('klein', 'flux-2-klein-9b-fp8.safetensors')
        assert keh.resolve_klein_unet('flux-2-klein-9b-fp8.safetensors') == \
            os.path.join('klein', 'flux-2-klein-9b-fp8.safetensors')


def test_resolve_vae_and_text_encoder_pick_installed_files(app, tmp_path):
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    with app.app_context():
        _comfy(tmp_path, cfg)
        assert keh.resolve_klein_vae() == 'flux2-vae.safetensors'
        assert keh.resolve_klein_text_encoder() == 'qwen_3_8b_fp8mixed.safetensors'


def test_resolve_text_encoder_never_grabs_another_familys_qwen(app, tmp_path):
    """Regression (live repro 2026-07-10): on a ComfyUI shared with other apps,
    text_encoders/ holds qwen3vl_4b (Z-Image) and qwen_2.5_vl (Qwen-Image) next
    to Klein's qwen_3_8b_fp8mixed. qwen3vl_4b sorts FIRST ('3' < '_'), and a
    loose "contains 'qwen'" match wired it into the graph -> KSampler died on
    'mat1 and mat2 shapes cannot be multiplied'. Canonical file present -> must
    win; only foreign qwen encoders present -> None (missing => auto-download),
    NEVER a wrong-family pick."""
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    with app.app_context():
        base = _comfy(tmp_path, cfg, te=True)
        _install(base, 'models', 'text_encoders', 'qwen3vl_4b_fp8_scaled.safetensors')
        _install(base, 'models', 'text_encoders', 'qwen_2.5_vl_7b_fp8_scaled.safetensors')
        _install(base, 'models', 'text_encoders', 'clip_l.safetensors')
        assert keh.resolve_klein_text_encoder() == 'qwen_3_8b_fp8mixed.safetensors'
        # Remove the canonical file: foreign qwen encoders must NOT be picked up.
        os.remove(str(base / 'models' / 'text_encoders' / 'qwen_3_8b_fp8mixed.safetensors'))
        assert keh.resolve_klein_text_encoder() is None
        assert 'klein_text_encoder' in keh.klein_missing_assets()


def test_resolve_vae_never_grabs_another_familys_vae(app, tmp_path):
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    with app.app_context():
        base = _comfy(tmp_path, cfg, vae=True)
        _install(base, 'models', 'vae', 'qwen_image_vae.safetensors')
        _install(base, 'models', 'vae', 'Wan2_2_VAE_bf16.safetensors')
        # The double-extension variant some installs carry is an acceptable flux2 match.
        _install(base, 'models', 'vae', 'flux2_vae.safetensors.safetensors')
        assert keh.resolve_klein_vae() == 'flux2-vae.safetensors'      # canonical wins
        os.remove(str(base / 'models' / 'vae' / 'flux2-vae.safetensors'))
        assert keh.resolve_klein_vae() == 'flux2_vae.safetensors.safetensors'  # narrow token ok
        os.remove(str(base / 'models' / 'vae' / 'flux2_vae.safetensors.safetensors'))
        assert keh.resolve_klein_vae() is None                          # never wan/qwen


def test_missing_assets_reports_absent_subset(app, tmp_path):
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    with app.app_context():
        _comfy(tmp_path, cfg, unet=True, vae=False, te=True, lora=False)
        missing = keh.klein_missing_assets()
        assert 'klein_vae' in missing and 'klein_lora' in missing
        assert 'klein_model' not in missing and 'klein_text_encoder' not in missing


def test_missing_assets_all_when_unconfigured(app):
    from app.services import klein_edit_helper as keh
    with app.app_context():
        missing = keh.klein_missing_assets()
        assert set(keh.KLEIN_REQUIRED).issubset(set(missing))


# --- Present-but-INVALID: the state between "missing" and "ready" ------------
def test_invalid_assets_flag_present_but_unloadable_file(app, tmp_path):
    """The #help incident at the resolver layer: the UNET file EXISTS (so it is NOT
    in klein_missing_assets) but its bytes are the HTML licence-gate page saved by a
    browser download without the licence. klein_invalid_assets names it as a
    distinct, actionable 'present but INVALID' state — not 'missing'."""
    from app import config as cfg
    from app.services import klein_edit_helper as keh, model_integrity
    with app.app_context():
        base = _comfy(tmp_path, cfg)   # all three present + valid
        _install(base, 'models', 'unet', 'klein', 'flux-2-klein-9b-fp8.safetensors',
                 data=b'<!doctype html><html>Access to this model is gated</html>')
        model_integrity.clear_cache()
        assert 'klein_model' not in keh.klein_missing_assets()      # it IS on disk
        inv = {i['asset']: i for i in keh.klein_invalid_assets()}
        assert 'klein_model' in inv
        assert inv['klein_model']['blocking'] is True
        assert inv['klein_model']['verdict'] == 'html_or_text'
        assert 'flux-2-klein-9b-fp8.safetensors' in inv['klein_model']['reason']


def test_invalid_assets_never_blocking_when_headers_valid(app, tmp_path):
    """Valid (if tiny) headers are never *blocking*-invalid — only the advisory
    too_small may appear for a stub, so nothing here blocks."""
    from app import config as cfg
    from app.services import klein_edit_helper as keh, model_integrity
    with app.app_context():
        _comfy(tmp_path, cfg, lora=True)
        model_integrity.clear_cache()
        assert all(not i['blocking'] for i in keh.klein_invalid_assets())


def test_invalid_assets_empty_when_unconfigured(app):
    from app.services import klein_edit_helper as keh
    with app.app_context():
        assert keh.klein_invalid_assets() == []


# --- Node 139 bypass is conditional on the base LoRA existing ---------------
def test_base_lora_node_kept_when_its_file_is_present(app, tmp_path, monkeypatch):
    """Inverse of the bypass test: when the workflow's base LoRA file DOES exist,
    node 139 stays in the graph (only bypassed when absent)."""
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    from app.job_queue import queue_manager
    with app.app_context():
        base = _comfy(tmp_path, cfg, lora=True)
        # Materialise the exact base LoRA the workflow references (node 139).
        _install(base, 'models', 'loras', 'klein', 'realistic.safetensors')
        cfg.save_config({'klein': {'consistency_lora': 'klein/Flux2-Klein-9B-consistency-V2.safetensors'}})
        src = tmp_path / 'ref.png'; src.write_bytes(_png())
        captured = {}
        monkeypatch.setattr(queue_manager, 'add_job', lambda **kw: (captured.update(kw), kw['job_id'])[1])
        keh.enqueue_klein_edit(user_id='local', source_filename='ref.png',
                               edit_prompt='p', source_path=str(src))
        wf = captured['workflow_data']
        assert '139' in wf   # base LoRA present -> not bypassed
        # The injected consistency node now carries a NUMERIC id — find it by title.
        cons_id = next(k for k, n in wf.items()
                       if (n.get('_meta') or {}).get('title') == 'Dataset consistency LoRA')
        assert cons_id.isdigit()
        assert wf['139']['inputs']['model'] == [cons_id, 0]


# --- Route: auto-download on missing models --------------------------------
def _make_ds(client):
    ds_id = client.post('/api/dataset/create',
                        json={'name': 'K', 'trigger_word': 'k'}).get_json()['id']
    client.post(f'/api/dataset/{ds_id}/ref',
                data={'file': (io.BytesIO(_png()), 'ref.png')},
                content_type='multipart/form-data')
    return ds_id


_KLEIN_BODY = {'generator': 'klein', 'multiplier': 1,
               'variations': [{'label': 'x', 'framing': 'face', 'prompt': 'p'}],
               'klein_model': 'flux-2-klein-9b-fp8.safetensors'}


def test_generate_missing_models_409_autostarts_public_not_gated(app, client, tmp_path, monkeypatch):
    from app import config as cfg, setup_installer
    started = []
    monkeypatch.setattr(setup_installer, 'start', lambda a: started.append(a))
    with app.app_context():
        _comfy(tmp_path, cfg, unet=False, vae=False, te=False, lora=False)  # valid ComfyUI, no models
    ds_id = _make_ds(client)
    resp = client.post(f'/api/dataset/{ds_id}/generate', json=_KLEIN_BODY)
    assert resp.status_code == 409
    body = resp.get_json()
    assert body['ok'] is False
    assert {'klein_model', 'klein_text_encoder', 'klein_vae'}.issubset(set(body['klein_missing']))
    assert body['needs_token'] is False                      # every Klein download is public now
    # the KV UNET is a public download → it fires without a token, like the others
    assert {'klein_model', 'klein_text_encoder', 'klein_vae'}.issubset(set(started))


def test_generate_missing_model_fires_without_token(app, client, tmp_path, monkeypatch):
    """The KV UNET is public: when it's the only missing asset it auto-downloads
    even with NO HF_TOKEN (the old gated behaviour refused to fire without one)."""
    from app import config as cfg, setup_installer
    started = []
    monkeypatch.setattr(setup_installer, 'start', lambda a: started.append(a))
    monkeypatch.delenv('HF_TOKEN', raising=False)
    with app.app_context():
        _comfy(tmp_path, cfg, unet=False, vae=True, te=True, lora=True)  # only the model missing
    ds_id = _make_ds(client)
    resp = client.post(f'/api/dataset/{ds_id}/generate', json=_KLEIN_BODY)
    assert resp.status_code == 409
    body = resp.get_json()
    assert body['needs_token'] is False
    assert started == ['klein_model']   # public → fires with no token


def test_generate_all_present_proceeds_and_bg_fetches_lora(app, client, tmp_path, monkeypatch):
    from app import config as cfg, setup_installer
    from app.job_queue import queue_manager
    started = []
    monkeypatch.setattr(setup_installer, 'start', lambda a: started.append(a))
    monkeypatch.setattr(queue_manager, 'add_job', lambda **kw: kw['job_id'])
    with app.app_context():
        _comfy(tmp_path, cfg, unet=True, vae=True, te=True, lora=False)  # required present, LoRA absent
    ds_id = _make_ds(client)
    resp = client.post(f'/api/dataset/{ds_id}/generate', json=_KLEIN_BODY)
    assert resp.status_code == 200
    assert resp.get_json()['created'] == 1
    # both optional LoRAs are fetched in the background: the consistency one, and the
    # improve detail LoRA whose absence silently disables the enhancement strength
    assert started == ['klein_lora', 'klein_enhancement_lora']


def test_default_consistency_strength_is_half(app):
    """The dx8152 consistency LoRA anchors STRUCTURE; its guide recommends
    starting at 0.5 and warns 0.8-1.0 can stop edits from applying. The old 0.9
    default made every variation a near-copy of the reference."""
    from app import config as cfg
    with app.app_context():
        assert cfg.get('klein.consistency_strength') == 0.5


def test_lora_strength_zero_skips_consistency_lora(app, tmp_path, monkeypatch):
    """Slider fully left (0) must disable the consistency LoRA entirely — the
    escape hatch when even mid strengths suppress the requested restaging."""
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    from app.job_queue import queue_manager
    with app.app_context():
        _comfy(tmp_path, cfg, lora=True)
        cfg.save_config({'klein': {'consistency_lora': 'klein/Flux2-Klein-9B-consistency-V2.safetensors'}})
        src = tmp_path / 'ref.png'; src.write_bytes(_png())
        captured = {}
        monkeypatch.setattr(queue_manager, 'add_job', lambda **kw: (captured.update(kw), kw['job_id'])[1])
        keh.enqueue_klein_edit(user_id='local', source_filename='ref.png',
                               edit_prompt='p', source_path=str(src), lora_strength=0)
        wf = captured['workflow_data']
        assert not any((n.get('_meta') or {}).get('title') == 'Dataset consistency LoRA'
                       for n in wf.values())
        # The improve detail LoRA is part of a complete Klein install now, so node 139
        # stays and keeps the workflow's own wiring — strength 0 disables the
        # CONSISTENCY LoRA (asserted above), not the detail one, which is a separate
        # setting. On a machine lacking realistic.safetensors, 139 is bypassed instead
        # and this reads ['114', 0] — covered by test_improve_params_reach_workflow.
        assert wf['102']['inputs']['model'] == ['139', 0]


def test_extra_refs_chain_native_reference_latents(app, tmp_path, monkeypatch):
    """Multi-reference: each extra identity ref becomes a Load->Scale->Encode->
    ReferenceLatent chain hanging off node 92, and the sampler's positive input
    points at the LAST link. No extras -> workflow untouched (77.positive == 92)."""
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    from app.job_queue import queue_manager
    with app.app_context():
        _comfy(tmp_path, cfg, lora=True)
        src = tmp_path / 'ref.png'; src.write_bytes(_png())
        r1 = tmp_path / 'extra1.webp'; r1.write_bytes(_png((10, 200, 10)))
        r2 = tmp_path / 'extra2.webp'; r2.write_bytes(_png((200, 10, 10)))
        captured = {}
        monkeypatch.setattr(queue_manager, 'add_job', lambda **kw: (captured.update(kw), kw['job_id'])[1])
        keh.enqueue_klein_edit(user_id='local', source_filename='ref.png',
                               edit_prompt='p', source_path=str(src),
                               extra_ref_paths=[str(r1), str(r2), str(tmp_path / 'gone.webp')])
        wf = captured['workflow_data']
        # Chain: 92 -> ds_ref1_latent -> ds_ref2_latent -> 77.positive (missing file skipped).
        assert wf['ds_ref1_latent']['inputs']['conditioning'] == ['92', 0]
        assert wf['ds_ref2_latent']['inputs']['conditioning'] == ['ds_ref1_latent', 0]
        assert wf['77']['inputs']['positive'] == ['ds_ref2_latent', 0]
        assert 'ds_ref3_latent' not in wf
        # Each extra encodes through the SAME VAE loader as the primary ref.
        assert wf['ds_ref1_encode']['inputs']['vae'] == ['10', 0]
        # Both extras were copied into the ComfyUI input dir.
        input_dir = os.path.join(str(tmp_path / 'comfyui'), 'input')
        assert any('extra1' in f for f in os.listdir(input_dir))
        assert any('extra2' in f for f in os.listdir(input_dir))

        # Control: no extras -> the sampler still reads the stock node 92.
        captured.clear()
        keh.enqueue_klein_edit(user_id='local', source_filename='ref.png',
                               edit_prompt='p', source_path=str(src))
        assert captured['workflow_data']['77']['inputs']['positive'] == ['92', 0]


def test_klein_fanout_passes_dataset_extra_refs(app, tmp_path, monkeypatch):
    """The fan-out forwards the dataset's extra refs (same files as the Nano
    Banana multi-ref path) into the Klein workflow."""
    import json as _json
    from app import config as cfg
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app.job_queue import queue_manager
    with app.app_context():
        _comfy(tmp_path, cfg, lora=True)
        ds = svc.create_dataset(LOCAL_USER, 'Multi', 'multi')
        d = svc._dataset_dir(ds.id)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'ref.webp'), 'wb') as fh:
            fh.write(_png())
        with open(os.path.join(d, 'xref.webp'), 'wb') as fh:
            fh.write(_png((5, 5, 250)))
        ds.ref_filename = 'ref.webp'
        ds.ref_extra_filenames = _json.dumps(['xref.webp'])
        svc.db.session.commit()
        captured = []
        monkeypatch.setattr(queue_manager, 'add_job',
                            lambda **kw: (captured.append(kw), kw['job_id'])[1])
        svc.generate_variations(LOCAL_USER, ds.id,
                                [{'label': 'x', 'framing': 'face', 'prompt': 'p'}], 1, None)
        wf = captured[0]['workflow_data']
        assert wf['77']['inputs']['positive'] == ['ds_ref1_latent', 0]


# --- Shared installs: klein models under diffusion_models/ -------------------
def test_scan_models_sees_klein_in_diffusion_models(app, tmp_path):
    """Shared ComfyUI installs keep e.g. diffusion_models/'Flux2 klein'/ (the KV
    variant) next to our canonical unet/klein/ download — both must feed the
    capability gate and the picker, deduped."""
    from app import config as cfg, capabilities
    with app.app_context():
        base = _comfy(tmp_path, cfg)     # unet/klein/flux-2-klein-9b-fp8
        _install(base, 'models', 'diffusion_models', 'Flux2 klein',
                 'flux-2-klein-9b-kv-fp8.safetensors')
        models = capabilities._scan_models()
        assert 'flux-2-klein-9b-fp8.safetensors' in models['klein']
        assert 'flux-2-klein-9b-kv-fp8.safetensors' in models['klein']


def test_resolver_honours_pick_from_diffusion_models_folder(app, tmp_path):
    """Picking the KV variant in the picker must resolve to its OWN subfolder
    prefix ('Flux2 klein\\...'), not be silently swapped for the unet/klein file."""
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    with app.app_context():
        base = _comfy(tmp_path, cfg)
        _install(base, 'models', 'diffusion_models', 'Flux2 klein',
                 'flux-2-klein-9b-kv-fp8.safetensors')
        assert keh.resolve_klein_unet('flux-2-klein-9b-kv-fp8.safetensors') == \
            os.path.join('Flux2 klein', 'flux-2-klein-9b-kv-fp8.safetensors')
        # No pick -> the canonical download wins; the canonical is now the KV build,
        # so it is preferred over a co-installed legacy (pre-KV) file.
        assert keh.resolve_klein_unet() == \
            os.path.join('Flux2 klein', 'flux-2-klein-9b-kv-fp8.safetensors')


# --- Flat / Stability-Matrix layout: klein model at the ROOT of a search folder ---
# waltm (Discord dev-chat) hit this: his flux-2-klein-9b-fp8.safetensors sat
# straight in diffusion_models/ with NO klein/ subfolder, so the scan (subfolders
# only) never bucketed it and the picker/resolver were both blind. The scan and the
# resolver must now BOTH see a root-level file — and stay in lockstep.
def test_scan_and_resolve_klein_at_diffusion_models_root(app, tmp_path):
    from app import config as cfg, capabilities
    from app.services import klein_edit_helper as keh
    with app.app_context():
        base = _comfy(tmp_path, cfg, unet=False)     # no unet/klein/ subfolder at all
        # Flat layout: the model file sits directly in diffusion_models/.
        _install(base, 'models', 'diffusion_models', 'flux-2-klein-9b-fp8.safetensors')
        # Picker: the scan lists the bare name.
        assert 'flux-2-klein-9b-fp8.safetensors' in capabilities._scan_models()['klein']
        # Resolver: a root-level file loads by its BARE name (no subfolder prefix) —
        # os.path.join('', name) == name, exactly what UNETLoader wants here.
        assert keh.resolve_klein_unet() == 'flux-2-klein-9b-fp8.safetensors'
        assert keh.resolve_klein_unet('flux-2-klein-9b-fp8.safetensors') == \
            'flux-2-klein-9b-fp8.safetensors'
        # Probe: the model is no longer "missing" for the generate preflight.
        assert 'klein_model' not in keh.klein_missing_assets()


def test_root_file_without_klein_in_name_is_ignored(app, tmp_path):
    """A non-Klein model dropped at the root of diffusion_models/ must NOT leak into
    the klein bucket or the resolver — the root scan buckets by name like the
    subfolder scan does."""
    from app import config as cfg, capabilities
    from app.services import klein_edit_helper as keh
    with app.app_context():
        base = _comfy(tmp_path, cfg, unet=False)
        _install(base, 'models', 'diffusion_models', 'some-random-sdxl.safetensors')
        assert capabilities._scan_models()['klein'] == []
        assert keh.resolve_klein_unet() is None
        assert 'klein_model' in keh.klein_missing_assets()


def test_klein_root_file_and_subfolder_both_discovered(app, tmp_path):
    """Mixed install: the canonical unet/klein/ download AND a flat root-level file
    under diffusion_models/. Both feed the picker (deduped), and a pick of each
    resolves to the RIGHT location — bare name for the root file, subfolder prefix
    for the subfolder file."""
    from app import config as cfg, capabilities
    from app.services import klein_edit_helper as keh
    with app.app_context():
        base = _comfy(tmp_path, cfg)                 # unet/klein/flux-2-klein-9b-fp8
        _install(base, 'models', 'diffusion_models', 'flux-2-klein-9b-kv-fp8.safetensors')
        klein = capabilities._scan_models()['klein']
        assert 'flux-2-klein-9b-fp8.safetensors' in klein      # subfolder file
        assert 'flux-2-klein-9b-kv-fp8.safetensors' in klein   # root file
        # The root pick loads bare; the subfolder pick keeps its prefix.
        assert keh.resolve_klein_unet('flux-2-klein-9b-kv-fp8.safetensors') == \
            'flux-2-klein-9b-kv-fp8.safetensors'
        assert keh.resolve_klein_unet('flux-2-klein-9b-fp8.safetensors') == \
            os.path.join('klein', 'flux-2-klein-9b-fp8.safetensors')


# --- NSFW mode (local Klein only) -------------------------------------------
def test_nsfw_catalog_entries_are_well_formed():
    from app.services.face_variations import NSFW_VARIATION_CATALOG, is_nsfw_label
    assert len(NSFW_VARIATION_CATALOG) >= 8
    for e in NSFW_VARIATION_CATALOG:
        assert e['id'].startswith('nsfw_')      # the UI's engine-switch cleanup keys on this
        assert e['framing'] in ('face', 'bust', 'body', 'back')
        assert is_nsfw_label(e['label'])
    assert is_nsfw_label('🔞 custom whatever')  # free-prompt marker
    assert not is_nsfw_label('Corps, plage (habille)')
    assert not is_nsfw_label(None)


def test_wrap_klein_nsfw_drops_sfw_clamp():
    from app.services.face_variations import wrap_variation_klein
    sfw = wrap_variation_klein('full body shot, standing')
    nsfw = wrap_variation_klein('full body shot, standing fully nude', nsfw=True)
    assert 'SFW' in sfw
    assert 'SFW' not in nsfw
    assert 'nudity is allowed' in nsfw
    # Identity constraint survives in both registers.
    assert 'Keep the facial identity exactly the same' in nsfw


def test_prompt_and_aspect_lookup_cover_nsfw_labels(app):
    from app.services.face_variations import prompt_by_label, aspect_for_label
    with app.app_context():
        assert 'nude' in (prompt_by_label('Corps, nu debout') or '')
        assert aspect_for_label('Corps, nu douche') == '9:16'


def test_generate_route_refuses_nsfw_on_api_engines(client):
    resp = client.post('/api/dataset/1/generate', json={
        'generator': 'nanobanana', 'multiplier': 1,
        'variations': [{'label': 'Corps, nu debout', 'framing': 'body',
                        'prompt': 'full body shot, standing fully nude'}],
    })
    assert resp.status_code == 400
    assert 'Klein' in resp.get_json()['error']


def test_service_fanout_refuses_nsfw_on_api_engines(app):
    import pytest
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'NoApi', 'noapi')
        with pytest.raises(ValueError, match='Klein engine only'):
            svc.generate_variations_nanobanana(
                None, LOCAL_USER, ds.id,
                [{'label': 'x', 'framing': 'body', 'prompt': 'p', 'nsfw': True}], 1)


def test_klein_fanout_nsfw_uses_uncensored_wrapper(app, tmp_path, monkeypatch):
    from app import config as cfg
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app.job_queue import queue_manager
    with app.app_context():
        _comfy(tmp_path, cfg, lora=True)
        ds = svc.create_dataset(LOCAL_USER, 'Nsfw', 'nsfw')
        d = svc._dataset_dir(ds.id)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'ref.webp'), 'wb') as fh:
            fh.write(_png())
        ds.ref_filename = 'ref.webp'
        svc.db.session.commit()
        captured = []
        monkeypatch.setattr(queue_manager, 'add_job',
                            lambda **kw: (captured.append(kw), kw['job_id'])[1])
        svc.generate_variations(LOCAL_USER, ds.id, [
            {'label': 'Corps, nu debout', 'framing': 'body',
             'prompt': 'full body shot, standing fully nude'},
            {'label': '🔞 custom', 'framing': 'body', 'prompt': 'custom pose', 'nsfw': True},
            {'label': 'Corps debout face', 'framing': 'body', 'prompt': 'full body shot'},
        ], 1, None)
        texts = [c['workflow_data']['6']['inputs']['text'] for c in captured]
        assert 'nudity is allowed' in texts[0] and 'SFW' not in texts[0]   # catalog label
        assert 'nudity is allowed' in texts[1]                             # explicit flag
        assert 'SFW' in texts[2]                                           # SFW entry untouched


def test_variations_route_ships_nsfw_catalog_separately(client):
    d = client.get('/api/dataset/variations').get_json()
    assert 'nsfw_catalog' in d and len(d['nsfw_catalog']) >= 8
    sfw_ids = {e['id'] for e in d['catalog']}
    assert not any(e['id'] in sfw_ids for e in d['nsfw_catalog'])
    # NSFW ids never leak into the presets (they are opt-in only).
    for ids in d['presets'].values():
        assert not any(i.startswith('nsfw_') for i in ids)


def test_wrap_klein_framing_detail_enriches_local_prompts():
    """Klein prompt study (fal.ai/BFL guides): Klein under-fills terse tag
    prompts (unlike the API engines which embellish on their own) — each framing
    injects a concrete full-intended-result description. API wrapper untouched."""
    from app.services.face_variations import wrap_variation, wrap_variation_klein
    body = wrap_variation_klein('full body shot, standing in a cafe', framing='body')
    assert 'head to toe' in body and '35mm' in body
    face = wrap_variation_klein('close-up portrait, smiling', framing='face')
    assert '85mm' in face and 'eyes in crisp focus' in face
    none = wrap_variation_klein('anything')          # unknown/absent framing -> no detail block
    assert '85mm' not in none and 'head to toe' not in none
    # The photographic tail is Klein-only steering (negatives are dead at CFG 1).
    assert 'natural skin texture' in body
    assert 'natural skin texture' not in wrap_variation('full body shot')


def test_wrap_variation_klein_is_instruction_first(app):
    """Klein is an instruction-edit model (Kontext lineage): the wrapper must ASK
    FOR THE CHANGE first and constrain the face second. The API-engine order
    (preserve-EXACTLY first, description after) made Klein return a near-copy of
    the reference — the live 'it just upscaled my reference' repro."""
    from app.services.face_variations import wrap_variation_klein
    p = wrap_variation_klein('close-up portrait, left profile view, neutral')
    assert 'close-up portrait, left profile view, neutral' in p
    change = p.index('Create a new photograph')
    compose = p.index('do not copy the composition')
    identity = p.index('Keep the facial identity')
    assert change < compose < identity
    assert 'Preserve their facial identity EXACTLY' not in p   # the API wrapper's poison


def test_klein_fanout_uses_instruction_wrapper(app, tmp_path, monkeypatch):
    """The Klein fan-out must feed the workflow's prompt node (6, CLIPTextEncode)
    the instruction-style wrapper, not the API engines' preservation-first guard."""
    from app import config as cfg
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app.job_queue import queue_manager
    with app.app_context():
        _comfy(tmp_path, cfg, lora=True)
        ds = svc.create_dataset(LOCAL_USER, 'Wrap', 'wrap')
        d = svc._dataset_dir(ds.id)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'ref.webp'), 'wb') as fh:
            fh.write(_png())
        ds.ref_filename = 'ref.webp'
        svc.db.session.commit()
        captured = []
        monkeypatch.setattr(queue_manager, 'add_job',
                            lambda **kw: (captured.append(kw), kw['job_id'])[1])
        svc.generate_variations(LOCAL_USER, ds.id,
                                [{'label': 'x', 'framing': 'face', 'prompt': 'left profile view'}],
                                1, klein_model=None)
        text = captured[0]['workflow_data']['6']['inputs']['text']
        assert 'left profile view' in text
        assert 'do not copy the composition' in text
        assert text.startswith('Create a new photograph')


def test_generate_unconfigured_comfyui_says_configure_first(app, client, tmp_path, monkeypatch):
    from app import setup_installer
    started = []
    monkeypatch.setattr(setup_installer, 'start', lambda a: started.append(a))
    ds_id = _make_ds(client)   # no comfyui configured at all
    resp = client.post(f'/api/dataset/{ds_id}/generate', json=_KLEIN_BODY)
    assert resp.status_code == 409
    assert 'ComfyUI install folder' in resp.get_json()['error']
    assert started == []   # nothing to download into -> nothing started


# --- Custom-node de-dependency + node preflight ------------------------------
# The shipped 'improve skin.json' historically needed two custom-node packs the
# Setup never installed (TextBox1 from RES4LYF, Any Switch (rgthree) from
# rgthree-comfy) -> a fresh install's FIRST generation died on a raw ComfyUI 400
# 'missing_node_type'. Strategy: (A) the workflow is rewritten to core +
# comfy_extras nodes only, and (B) the generate routes preflight the workflow's
# class_types against /object_info (fail-open) so any future/reverted custom
# node surfaces as one actionable 409.

def _shipped_classes(keh):
    import json
    with open(str(keh.WORKFLOW_IMPROVE_SKIN_PATH), encoding='utf-8') as f:
        return {n['class_type'] for n in json.load(f).values()}


def test_workflow_uses_only_core_nodes_and_is_fully_linked():
    """(A) No custom-node class_type may remain in improve skin.json, every link
    must reference an existing node, and no node may be orphaned. Pins the
    de-dependency so a re-export from the developer's ComfyUI can't silently
    reintroduce RES4LYF / rgthree nodes."""
    import json
    from app.services import klein_edit_helper as keh
    with open(str(keh.WORKFLOW_IMPROVE_SKIN_PATH), encoding='utf-8') as f:
        wf = json.load(f)
    classes = {n['class_type'] for n in wf.values()}
    assert not classes & set(keh.KLEIN_NODE_PACKS), classes & set(keh.KLEIN_NODE_PACKS)
    ids = set(wf)
    referenced = set()
    for nid, node in wf.items():
        for v in node.get('inputs', {}).values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                assert v[0] in ids, f'node {nid} references missing node {v[0]}'
                referenced.add(v[0])
    orphans = [nid for nid, n in wf.items()
               if nid not in referenced and n['class_type'] != 'SaveImage']
    assert orphans == [], orphans
    # The prompt now lives in the CLIPTextEncode widget itself (a literal string,
    # not a link to the removed TextBox1), and the latent/sampling size wiring
    # reads the PrimitiveInt nodes directly (rgthree switches removed).
    assert wf['6']['class_type'] == 'CLIPTextEncode'
    assert isinstance(wf['6']['inputs']['text'], str)
    for nid in ('91', '102'):
        assert wf[nid]['inputs']['width'] == ['119', 0]
        assert wf[nid]['inputs']['height'] == ['120', 0]


def test_klein_missing_nodes_maps_pack_and_url(app, monkeypatch):
    """(B) A class_type absent from /object_info is reported with the pack that
    ships it + the GitHub link (known packs), or pack/url None (unknown)."""
    from app.services import klein_edit_helper as keh
    with app.app_context():
        monkeypatch.setattr('app.utils.comfyui.fetch_object_info_classes',
                            lambda *a, **k: {'CLIPTextEncode'})
        wf = {'1': {'class_type': 'TextBox1', 'inputs': {}},
              '2': {'class_type': 'SomeOtherCustomNode', 'inputs': {}},
              '3': {'class_type': 'CLIPTextEncode', 'inputs': {}}}
        out = keh.klein_missing_nodes(wf)
        assert out == [
            {'class_type': 'SomeOtherCustomNode', 'pack': None, 'url': None},
            {'class_type': 'TextBox1', 'pack': 'RES4LYF',
             'url': 'https://github.com/ClownsharkBatwing/RES4LYF'},
        ]
        msg = keh.format_missing_nodes_message(out)
        assert 'RES4LYF' in msg and 'github.com/ClownsharkBatwing/RES4LYF' in msg
        assert 'ComfyUI-Manager' in msg and 'restart ComfyUI' in msg


def test_klein_missing_nodes_fails_open_when_object_info_down(app, monkeypatch):
    """(B) /object_info unreachable (fetch returns None) must NEVER block
    generation — returns [] (the per-job ComfyUI error still surfaces later)."""
    from app.services import klein_edit_helper as keh
    with app.app_context():
        monkeypatch.setattr('app.utils.comfyui.fetch_object_info_classes',
                            lambda *a, **k: None)
        assert keh.klein_missing_nodes({'1': {'class_type': 'TextBox1', 'inputs': {}}}) == []
        monkeypatch.setattr(keh, '_nodes_ok_until', 0.0)
        assert keh.klein_missing_nodes() == []   # shipped-workflow path too


def test_klein_missing_nodes_caches_only_the_all_present_verdict(app, monkeypatch):
    """The multi-MB /object_info probe is cached ONLY when everything is present:
    a satisfied install stops re-fetching per tile, while a missing-node verdict
    keeps re-probing so 'install the pack, restart, retry' works immediately."""
    from app.services import klein_edit_helper as keh
    calls = []
    with app.app_context():
        # All present -> verdict cached -> second call doesn't re-fetch.
        monkeypatch.setattr(keh, '_nodes_ok_until', 0.0)
        monkeypatch.setattr('app.utils.comfyui.fetch_object_info_classes',
                            lambda *a, **k: (calls.append(1), _shipped_classes(keh))[1])
        assert keh.klein_missing_nodes() == []
        assert keh.klein_missing_nodes() == []
        assert len(calls) == 1
        # Missing -> NOT cached -> every call re-probes.
        calls.clear()
        monkeypatch.setattr(keh, '_nodes_ok_until', 0.0)
        monkeypatch.setattr('app.utils.comfyui.fetch_object_info_classes',
                            lambda *a, **k: (calls.append(1), {'SaveImage'})[1])
        assert keh.klein_missing_nodes() != []
        assert keh.klein_missing_nodes() != []
        assert len(calls) == 2


def test_generate_missing_nodes_409_names_pack_and_link(app, client, tmp_path, monkeypatch):
    """(B) Route contract: models all present but the workflow needs a custom node
    this ComfyUI lacks -> ONE 409 through the existing Klein-missing handler, with
    the itemized `klein_nodes_missing` payload and an error string naming the pack
    + GitHub link + the ComfyUI-Manager instruction (rendered by the existing
    toast, no new frontend component)."""
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    with app.app_context():
        _comfy(tmp_path, cfg, lora=True)   # every model file present
    # Simulate a reverted/edited workflow that still carries TextBox1.
    monkeypatch.setattr(keh, 'load_workflow_local',
                        lambda p: {'145': {'class_type': 'TextBox1', 'inputs': {'text1': ''}},
                                   '9': {'class_type': 'SaveImage', 'inputs': {}}})
    monkeypatch.setattr(keh, '_nodes_ok_until', 0.0)
    monkeypatch.setattr('app.utils.comfyui.fetch_object_info_classes',
                        lambda *a, **k: {'SaveImage'})
    ds_id = _make_ds(client)
    resp = client.post(f'/api/dataset/{ds_id}/generate', json=_KLEIN_BODY)
    assert resp.status_code == 409
    body = resp.get_json()
    assert body['ok'] is False
    assert body['klein_nodes_missing'] == [
        {'class_type': 'TextBox1', 'pack': 'RES4LYF',
         'url': 'https://github.com/ClownsharkBatwing/RES4LYF'}]
    assert body['klein_missing'] == []              # models are NOT the problem
    assert 'RES4LYF' in body['error'] and 'ComfyUI-Manager' in body['error']
    # No doomed rows were created (the preflight blocked before the fan-out).
    payload = client.get(f'/api/dataset/{ds_id}').get_json()
    assert payload['images'] == []


def test_generate_proceeds_when_object_info_unreachable(app, client, tmp_path, monkeypatch):
    """(B) Fail-open at the route: /object_info down + all models present must
    still enqueue (never block on a transient probe failure)."""
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    from app.job_queue import queue_manager
    monkeypatch.setattr(queue_manager, 'add_job', lambda **kw: kw['job_id'])
    with app.app_context():
        _comfy(tmp_path, cfg, lora=True)
    monkeypatch.setattr(keh, '_nodes_ok_until', 0.0)
    monkeypatch.setattr('app.utils.comfyui.fetch_object_info_classes',
                        lambda *a, **k: None)
    ds_id = _make_ds(client)
    resp = client.post(f'/api/dataset/{ds_id}/generate', json=_KLEIN_BODY)
    assert resp.status_code == 200
    assert resp.get_json()['created'] == 1


def test_regenerate_missing_nodes_409(app, client, tmp_path, monkeypatch):
    """(B) The single-tile regenerate path preflights nodes too (same 409 shape),
    instead of resetting the row to pending and letting ComfyUI 400 it."""
    from app import config as cfg
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper as keh
    from app.config import LOCAL_USER
    with app.app_context():
        _comfy(tmp_path, cfg, lora=True)
        ds = svc.create_dataset(LOCAL_USER, 'Regen', 'regen')
        d = svc._dataset_dir(ds.id)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'ref.webp'), 'wb') as fh:
            fh.write(_png())
        ds.ref_filename = 'ref.webp'
        img = svc.FaceDatasetImage(dataset_id=ds.id, source='generated',
                                   status='finished', variation_prompt='p')
        svc.db.session.add(img)
        svc.db.session.commit()
        img_id = img.id
    monkeypatch.setattr(keh, 'load_workflow_local',
                        lambda p: {'145': {'class_type': 'TextBox1', 'inputs': {'text1': ''}}})
    monkeypatch.setattr(keh, '_nodes_ok_until', 0.0)
    monkeypatch.setattr('app.utils.comfyui.fetch_object_info_classes',
                        lambda *a, **k: {'SaveImage'})
    resp = client.post(f'/api/dataset/image/{img_id}/regenerate', json={})
    assert resp.status_code == 409
    body = resp.get_json()
    assert body['klein_nodes_missing'][0]['pack'] == 'RES4LYF'
    with app.app_context():
        row = svc.db.session.get(svc.FaceDatasetImage, img_id)
        assert row.status == 'finished'   # untouched — blocked before any reset
