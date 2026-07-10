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


def test_generate_unconfigured_comfyui_says_configure_first(app, client, tmp_path, monkeypatch):
    from app import setup_installer
    started = []
    monkeypatch.setattr(setup_installer, 'start', lambda a: started.append(a))
    ds_id = _make_ds(client)   # no comfyui configured at all
    resp = client.post(f'/api/dataset/{ds_id}/generate', json=_KLEIN_BODY)
    assert resp.status_code == 409
    assert 'ComfyUI install folder' in resp.get_json()['error']
    assert started == []   # nothing to download into -> nothing started
