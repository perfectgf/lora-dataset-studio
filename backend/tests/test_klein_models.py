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

import pytest
from PIL import Image


def _png(color=(0, 128, 255)):
    buf = io.BytesIO(); Image.new('RGB', (64, 64), color).save(buf, 'PNG')
    return buf.getvalue()


def _install(base, *relparts, data=b'x'):
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
        assert wf['139']['inputs']['model'] == ['ds_consistency_lora', 0]


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
    assert body['needs_token'] is True                       # gated model + no HF_TOKEN
    assert 'klein_text_encoder' in started and 'klein_vae' in started
    assert 'klein_model' not in started                      # gated -> not fired without a token


def test_generate_missing_gated_model_fires_when_token_present(app, client, tmp_path, monkeypatch):
    from app import config as cfg, setup_installer
    started = []
    monkeypatch.setattr(setup_installer, 'start', lambda a: started.append(a))
    monkeypatch.setenv('HF_TOKEN', 'hf_token')
    with app.app_context():
        _comfy(tmp_path, cfg, unet=False, vae=True, te=True, lora=True)  # only the model missing
    ds_id = _make_ds(client)
    resp = client.post(f'/api/dataset/{ds_id}/generate', json=_KLEIN_BODY)
    assert resp.status_code == 409
    body = resp.get_json()
    assert body['needs_token'] is False
    assert started == ['klein_model']   # token present -> the gated download IS fired


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
    assert started == ['klein_lora']   # optional consistency LoRA fetched in the background


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
        assert 'ds_consistency_lora' not in wf
        # Base 'realistic' LoRA absent too -> node 139 bypassed -> 102 wired to the UNET.
        assert wf['102']['inputs']['model'] == ['114', 0]


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
        # No pick -> canonical download still wins.
        assert keh.resolve_klein_unet() == os.path.join('klein', 'flux-2-klein-9b-fp8.safetensors')


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
        texts = [c['workflow_data']['145']['inputs']['text1'] for c in captured]
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
    """The Klein fan-out must feed the workflow's prompt node (145) the
    instruction-style wrapper, not the API engines' preservation-first guard."""
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
        text = captured[0]['workflow_data']['145']['inputs']['text1']
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
