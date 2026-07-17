"""Optional generation LoRAs (Idea by @waltm — Discord feature request).

Two opt-in, USER-POINTED LoRA slots chained onto the local Klein edit graph
after the consistency LoRA: `ultra_real` (skin/texture, SFW+NSFW) and
`nsfw_anatomy` (STRICTLY gated behind the run's NSFW toggle). These tests pin:
(a) the graph wiring order 114 -> consistency -> ultra_real -> nsfw -> 139,
(b) silent degradation (missing file / empty config / strength 0 / None),
(c) the NSFW gate at the service level (SFW shots of a mixed batch never get
    the anatomy LoRA; regenerate keys on the tile's own label),
(d) the config round-trip through /api/settings.
"""
import io
import os

from PIL import Image


def _png(color=(0, 128, 255)):
    buf = io.BytesIO(); Image.new('RGB', (64, 64), color).save(buf, 'PNG')
    return buf.getvalue()


def _install(base, *relparts, data=b'x'):
    p = base.joinpath(*relparts)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def _comfy(tmp_path, cfg, ultra=True, nsfw=True, consistency=True, base_lora=False):
    """A ComfyUI tree with all REQUIRED Klein assets and a configurable subset of
    the optional LoRA files. Configures the two optional slots to point at
    klein/ultra-real-test.safetensors / klein/nsfw-anatomy-test.safetensors
    whether or not the files exist (degradation is the point)."""
    base = tmp_path / 'comfyui'
    (base / 'input').mkdir(parents=True)
    (base / 'output').mkdir(parents=True)
    (base / 'main.py').write_text('# fake', encoding='utf-8')
    _install(base, 'models', 'unet', 'klein', 'flux-2-klein-9b-fp8.safetensors')
    _install(base, 'models', 'vae', 'flux2-vae.safetensors')
    _install(base, 'models', 'text_encoders', 'qwen_3_8b_fp8mixed.safetensors')
    if consistency:
        _install(base, 'models', 'loras', 'klein', 'Flux2-Klein-9B-consistency-V2.safetensors')
    if ultra:
        _install(base, 'models', 'loras', 'klein', 'ultra-real-test.safetensors')
    if nsfw:
        _install(base, 'models', 'loras', 'klein', 'nsfw-anatomy-test.safetensors')
    if base_lora:
        _install(base, 'models', 'loras', 'klein', 'realistic.safetensors')
    cfg.save_config({'comfyui': {'base_dir': str(base)},
                     'klein': {'ultra_real_lora': 'klein/ultra-real-test.safetensors',
                               'nsfw_lora': 'klein/nsfw-anatomy-test.safetensors'}})
    return base


def _enqueue(keh, queue_manager, monkeypatch, tmp_path, **kwargs):
    src = tmp_path / 'ref.png'
    if not src.exists():
        src.write_bytes(_png())
    captured = {}
    monkeypatch.setattr(queue_manager, 'add_job',
                        lambda **kw: (captured.update(kw), kw['job_id'])[1])
    keh.enqueue_klein_edit(user_id='local', source_filename='ref.png',
                           edit_prompt='p', source_path=str(src), **kwargs)
    return captured['workflow_data']


# --- Defaults ---------------------------------------------------------------
def test_config_defaults_no_hardcoded_lora_names(app):
    """The slots ship EMPTY: the user points the files (never a name we invent —
    the shipped workflow's own hardcoded LoRA names are exactly what
    klein_edit_helper exists to undo)."""
    from app import config as cfg
    with app.app_context():
        assert cfg.get('klein.ultra_real_lora') == ''
        assert cfg.get('klein.nsfw_lora') == ''
        assert cfg.get('klein.ultra_real_strength') == 0.6
        assert cfg.get('klein.nsfw_strength') == 0.6


def test_config_roundtrip_through_settings_api(app, client):
    resp = client.put('/api/settings', json={'config': {'klein': {
        'ultra_real_lora': 'klein/my-texture.safetensors', 'ultra_real_strength': 0.8,
        'nsfw_lora': 'klein/my-anatomy.safetensors', 'nsfw_strength': 1.0}}})
    assert resp.status_code == 200
    saved = client.get('/api/settings').get_json()['config']['klein']
    assert saved['ultra_real_lora'] == 'klein/my-texture.safetensors'
    assert saved['ultra_real_strength'] == 0.8
    assert saved['nsfw_lora'] == 'klein/my-anatomy.safetensors'
    assert saved['nsfw_strength'] == 1.0
    # The untouched neighbours survive the partial save.
    assert saved['consistency_strength'] == 0.5


# --- Graph wiring ------------------------------------------------------------
def test_full_chain_order_consistency_then_ultra_then_nsfw(app, tmp_path, monkeypatch):
    """114 -> consistency -> ultra_real -> nsfw_anatomy -> 139 (base LoRA file
    present so node 139 stays), each link a LoraLoaderModelOnly hanging off the
    previous one."""
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    from app.job_queue import queue_manager
    with app.app_context():
        _comfy(tmp_path, cfg, base_lora=True)
        wf = _enqueue(keh, queue_manager, monkeypatch, tmp_path,
                      ultra_real_strength=0.7, nsfw_lora_strength=0.9)
        assert wf['ds_consistency_lora']['inputs']['model'] == ['114', 0]
        assert wf['ds_ultra_real_lora']['inputs']['model'] == ['ds_consistency_lora', 0]
        assert wf['ds_nsfw_lora']['inputs']['model'] == ['ds_ultra_real_lora', 0]
        assert wf['139']['inputs']['model'] == ['ds_nsfw_lora', 0]
        assert wf['ds_ultra_real_lora']['class_type'] == 'LoraLoaderModelOnly'
        assert wf['ds_ultra_real_lora']['inputs']['lora_name'] == \
            os.path.join('klein', 'ultra-real-test.safetensors')
        assert wf['ds_ultra_real_lora']['inputs']['strength_model'] == 0.7
        assert wf['ds_nsfw_lora']['inputs']['lora_name'] == \
            os.path.join('klein', 'nsfw-anatomy-test.safetensors')
        assert wf['ds_nsfw_lora']['inputs']['strength_model'] == 0.9


def test_chain_survives_base_lora_bypass(app, tmp_path, monkeypatch):
    """Base 'realistic' LoRA absent -> node 139 bypassed -> 102 wires straight to
    the LAST optional LoRA (the chain stays intact)."""
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    from app.job_queue import queue_manager
    with app.app_context():
        _comfy(tmp_path, cfg)          # no realistic.safetensors on disk
        wf = _enqueue(keh, queue_manager, monkeypatch, tmp_path,
                      ultra_real_strength=0.7, nsfw_lora_strength=0.9)
        assert '139' not in wf
        assert wf['102']['inputs']['model'] == ['ds_nsfw_lora', 0]
        assert wf['ds_nsfw_lora']['inputs']['model'] == ['ds_ultra_real_lora', 0]


def test_ultra_real_alone_hangs_off_consistency(app, tmp_path, monkeypatch):
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    from app.job_queue import queue_manager
    with app.app_context():
        _comfy(tmp_path, cfg)
        wf = _enqueue(keh, queue_manager, monkeypatch, tmp_path,
                      ultra_real_strength=0.7)
        assert wf['ds_ultra_real_lora']['inputs']['model'] == ['ds_consistency_lora', 0]
        assert 'ds_nsfw_lora' not in wf
        assert wf['102']['inputs']['model'] == ['ds_ultra_real_lora', 0]


def test_slots_hang_off_unet_when_consistency_is_off(app, tmp_path, monkeypatch):
    """Consistency slider at 0 (skipped) -> the optional chain starts at the
    UNET (114) directly."""
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    from app.job_queue import queue_manager
    with app.app_context():
        _comfy(tmp_path, cfg)
        wf = _enqueue(keh, queue_manager, monkeypatch, tmp_path,
                      lora_strength=0, ultra_real_strength=0.7)
        assert 'ds_consistency_lora' not in wf
        assert wf['ds_ultra_real_lora']['inputs']['model'] == ['114', 0]
        assert wf['102']['inputs']['model'] == ['ds_ultra_real_lora', 0]


def test_strengths_clamped_to_1_5(app, tmp_path, monkeypatch):
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    from app.job_queue import queue_manager
    with app.app_context():
        _comfy(tmp_path, cfg)
        wf = _enqueue(keh, queue_manager, monkeypatch, tmp_path,
                      ultra_real_strength=9, nsfw_lora_strength=2)
        assert wf['ds_ultra_real_lora']['inputs']['strength_model'] == 1.5
        assert wf['ds_nsfw_lora']['inputs']['strength_model'] == 1.5


# --- Silent degradation ------------------------------------------------------
def test_none_strength_means_slot_off(app, tmp_path, monkeypatch):
    """Files configured AND on disk, but no per-run strength -> nothing injected
    (a configured file alone never arms a slot)."""
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    from app.job_queue import queue_manager
    with app.app_context():
        _comfy(tmp_path, cfg)
        wf = _enqueue(keh, queue_manager, monkeypatch, tmp_path)
        assert 'ds_ultra_real_lora' not in wf
        assert 'ds_nsfw_lora' not in wf
        assert wf['102']['inputs']['model'] == ['ds_consistency_lora', 0]


def test_strength_zero_skips_slot(app, tmp_path, monkeypatch):
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    from app.job_queue import queue_manager
    with app.app_context():
        _comfy(tmp_path, cfg)
        wf = _enqueue(keh, queue_manager, monkeypatch, tmp_path,
                      ultra_real_strength=0, nsfw_lora_strength=0)
        assert 'ds_ultra_real_lora' not in wf
        assert 'ds_nsfw_lora' not in wf


def test_missing_file_degrades_silently(app, tmp_path, monkeypatch):
    """Configured names whose files are NOT on disk -> both slots skipped, the
    job still enqueues (mirrors the consistency degradation, never a doomed
    ComfyUI validation error)."""
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    from app.job_queue import queue_manager
    with app.app_context():
        _comfy(tmp_path, cfg, ultra=False, nsfw=False)
        wf = _enqueue(keh, queue_manager, monkeypatch, tmp_path,
                      ultra_real_strength=0.7, nsfw_lora_strength=0.9)
        assert 'ds_ultra_real_lora' not in wf
        assert 'ds_nsfw_lora' not in wf
        assert wf['102']['inputs']['model'] == ['ds_consistency_lora', 0]


def test_unconfigured_slot_degrades_silently(app, tmp_path, monkeypatch):
    """Empty config names (the shipped default) -> strengths are ignored."""
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    from app.job_queue import queue_manager
    with app.app_context():
        _comfy(tmp_path, cfg)
        cfg.save_config({'klein': {'ultra_real_lora': '', 'nsfw_lora': ''}})
        wf = _enqueue(keh, queue_manager, monkeypatch, tmp_path,
                      ultra_real_strength=0.7, nsfw_lora_strength=0.9)
        assert 'ds_ultra_real_lora' not in wf
        assert 'ds_nsfw_lora' not in wf


# --- NSFW gate at the service level ------------------------------------------
def _dataset_with_ref(svc, LOCAL_USER):
    ds = svc.create_dataset(LOCAL_USER, 'Loras', 'loras')
    d = svc._dataset_dir(ds.id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'ref.webp'), 'wb') as fh:
        fh.write(_png())
    ds.ref_filename = 'ref.webp'
    svc.db.session.commit()
    return ds


def test_fanout_gates_nsfw_lora_per_variation(app, tmp_path, monkeypatch):
    """Mixed batch with the NSFW toggle on: the anatomy LoRA rides ONLY the NSFW
    variations; ultra_real rides every one."""
    from app import config as cfg
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app.job_queue import queue_manager
    with app.app_context():
        _comfy(tmp_path, cfg)
        ds = _dataset_with_ref(svc, LOCAL_USER)
        captured = []
        monkeypatch.setattr(queue_manager, 'add_job',
                            lambda **kw: (captured.append(kw), kw['job_id'])[1])
        svc.generate_variations(LOCAL_USER, ds.id, [
            {'label': 'Corps, nu debout', 'framing': 'body', 'prompt': 'nude'},
            {'label': '🔞 custom', 'framing': 'body', 'prompt': 'pose', 'nsfw': True},
            {'label': 'Corps debout face', 'framing': 'body', 'prompt': 'standing'},
        ], 1, None, ultra_real_strength=0.7, nsfw_lora_strength=0.9)
        wfs = [c['workflow_data'] for c in captured]
        assert all('ds_ultra_real_lora' in wf for wf in wfs)
        assert 'ds_nsfw_lora' in wfs[0]          # NSFW catalog label
        assert 'ds_nsfw_lora' in wfs[1]          # explicit nsfw flag (🔞 custom)
        assert 'ds_nsfw_lora' not in wfs[2]      # SFW shot of the same batch


def test_generate_route_passes_slot_strengths(app, client, tmp_path, monkeypatch):
    from app import config as cfg
    from app.job_queue import queue_manager
    captured = []
    monkeypatch.setattr(queue_manager, 'add_job',
                        lambda **kw: (captured.append(kw), kw['job_id'])[1])
    with app.app_context():
        _comfy(tmp_path, cfg)
    ds_id = client.post('/api/dataset/create',
                        json={'name': 'R', 'trigger_word': 'r'}).get_json()['id']
    client.post(f'/api/dataset/{ds_id}/ref',
                data={'file': (io.BytesIO(_png()), 'ref.png')},
                content_type='multipart/form-data')
    resp = client.post(f'/api/dataset/{ds_id}/generate', json={
        'generator': 'klein', 'multiplier': 1,
        'variations': [{'label': 'Corps, nu debout', 'framing': 'body', 'prompt': 'p'}],
        'ultra_real_strength': 0.7, 'nsfw_lora_strength': 0.9})
    assert resp.status_code == 200
    wf = captured[0]['workflow_data']
    assert 'ds_ultra_real_lora' in wf and 'ds_nsfw_lora' in wf


def test_generate_route_without_slot_keys_keeps_slots_off(app, client, tmp_path, monkeypatch):
    """Legacy/omitted body keys (the default UI state) -> no optional LoRA nodes
    even with both files configured and present."""
    from app import config as cfg
    from app.job_queue import queue_manager
    captured = []
    monkeypatch.setattr(queue_manager, 'add_job',
                        lambda **kw: (captured.append(kw), kw['job_id'])[1])
    with app.app_context():
        _comfy(tmp_path, cfg)
    ds_id = client.post('/api/dataset/create',
                        json={'name': 'Off', 'trigger_word': 'off'}).get_json()['id']
    client.post(f'/api/dataset/{ds_id}/ref',
                data={'file': (io.BytesIO(_png()), 'ref.png')},
                content_type='multipart/form-data')
    resp = client.post(f'/api/dataset/{ds_id}/generate', json={
        'generator': 'klein', 'multiplier': 1,
        'variations': [{'label': 'x', 'framing': 'face', 'prompt': 'p'}]})
    assert resp.status_code == 200
    wf = captured[0]['workflow_data']
    assert 'ds_ultra_real_lora' not in wf and 'ds_nsfw_lora' not in wf


def test_regenerate_gates_nsfw_lora_on_the_tiles_label(app, tmp_path, monkeypatch):
    """Regenerate: an SFW tile never receives the anatomy LoRA even when the
    caller sends a strength; an NSFW-labelled tile does."""
    from app import config as cfg
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app.job_queue import queue_manager
    with app.app_context():
        _comfy(tmp_path, cfg)
        ds = _dataset_with_ref(svc, LOCAL_USER)
        rows = {}
        for label in ('Corps debout face', 'Corps, nu debout'):
            img = svc.FaceDatasetImage(dataset_id=ds.id, source='generated',
                                       status='finished', variation_label=label,
                                       variation_prompt='p', framing='body')
            svc.db.session.add(img)
            svc.db.session.commit()
            rows[label] = img.id
        captured = []
        monkeypatch.setattr(queue_manager, 'add_job',
                            lambda **kw: (captured.append(kw), kw['job_id'])[1])
        svc.regenerate_image(LOCAL_USER, rows['Corps debout face'],
                             ultra_real_strength=0.7, nsfw_lora_strength=0.9)
        svc.regenerate_image(LOCAL_USER, rows['Corps, nu debout'],
                             ultra_real_strength=0.7, nsfw_lora_strength=0.9)
        sfw_wf, nsfw_wf = captured[0]['workflow_data'], captured[1]['workflow_data']
        assert 'ds_ultra_real_lora' in sfw_wf and 'ds_nsfw_lora' not in sfw_wf
        assert 'ds_ultra_real_lora' in nsfw_wf and 'ds_nsfw_lora' in nsfw_wf
