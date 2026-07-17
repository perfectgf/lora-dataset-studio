"""Optional generation LoRAs (Idea by @waltm — Discord feature request).

An ORDERED, user-defined list (`klein.generation_loras`, entries
{file, strength, nsfw_only}) of extra LoRAs chained onto the local Klein edit
graph after the consistency LoRA, in LIST order. These tests pin:
(a) the graph wiring 114 -> consistency -> gen_1 -> ... -> gen_N -> 139 for
    N > 2, order preserved, cap at MAX_GENERATION_LORAS;
(b) the soft migration of the short-lived single-slot keys
    (ultra_real_lora / nsfw_lora) into the list (nsfw -> nsfw_only=True),
    idempotent, legacy keys dropped, purged from the file on save;
(c) per-row silent degradation (missing file / blank / strength 0) with the
    rest of the chain still linking up;
(d) the per-row nsfw_only gate at the service level (SFW shots of a mixed
    batch never get an nsfw_only row; regenerate keys on the tile's label);
(e) requests can only arm CONFIGURED files (order + nsfw_only come from the
    config, never from the request) and the config round-trip via /api/settings.
"""
import io
import json
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


# Three ordered stand-in generation LoRAs; the middle one is flagged nsfw_only.
GEN_LORAS = [
    {'file': 'klein/gen-a.safetensors', 'strength': 0.6, 'nsfw_only': False},
    {'file': 'klein/gen-b-hot.safetensors', 'strength': 0.8, 'nsfw_only': True},
    {'file': 'klein/gen-c.safetensors', 'strength': 0.5, 'nsfw_only': False},
]
# The per-run request arming all three rows (what the UI sends).
ARM_ALL = [{'file': e['file'], 'strength': e['strength']} for e in GEN_LORAS]


def _comfy(tmp_path, cfg, gen_files=('gen-a', 'gen-b-hot', 'gen-c'),
           base_lora=False, config_loras=GEN_LORAS):
    """A ComfyUI tree with all REQUIRED Klein assets + the consistency LoRA, a
    configurable subset of the generation-LoRA FILES on disk, and the
    generation_loras LIST configured (files on disk or not — degradation is
    the point)."""
    base = tmp_path / 'comfyui'
    (base / 'input').mkdir(parents=True)
    (base / 'output').mkdir(parents=True)
    (base / 'main.py').write_text('# fake', encoding='utf-8')
    _install(base, 'models', 'unet', 'klein', 'flux-2-klein-9b-fp8.safetensors')
    _install(base, 'models', 'vae', 'flux2-vae.safetensors')
    _install(base, 'models', 'text_encoders', 'qwen_3_8b_fp8mixed.safetensors')
    _install(base, 'models', 'loras', 'klein', 'Flux2-Klein-9B-consistency-V2.safetensors')
    for stem in gen_files:
        _install(base, 'models', 'loras', 'klein', f'{stem}.safetensors')
    if base_lora:
        _install(base, 'models', 'loras', 'klein', 'realistic.safetensors')
    cfg.save_config({'comfyui': {'base_dir': str(base)},
                     'klein': {'generation_loras': config_loras}})
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


# --- Defaults & migration ----------------------------------------------------
def test_config_defaults_empty_list_no_hardcoded_names(app):
    """The list ships EMPTY: the user points every file (never a name we
    invent) and the legacy single-slot keys are gone from the defaults."""
    from app import config as cfg
    with app.app_context():
        assert cfg.get('klein.generation_loras') == []
        assert cfg.get('klein.ultra_real_lora') is None
        assert cfg.get('klein.nsfw_lora') is None


def test_legacy_single_slot_keys_migrate_into_the_list(app, tmp_path):
    """A config.json written by the single-slot version (an hour old!) keeps
    working: non-empty legacy keys become ordered list entries (ultra first,
    nsfw second with nsfw_only=True, strengths carried over) and the legacy
    keys are dropped from the loaded config."""
    from app import config as cfg
    with app.app_context():
        p = os.environ['LDS_CONFIG']
        with open(p, 'w', encoding='utf-8') as fh:
            json.dump({'klein': {'ultra_real_lora': 'klein/tex.safetensors',
                                 'ultra_real_strength': 0.7,
                                 'nsfw_lora': 'klein/hot.safetensors',
                                 'nsfw_strength': 1.0}}, fh)
        conf = cfg.load_config(force=True)
        assert conf['klein']['generation_loras'] == [
            {'file': 'klein/tex.safetensors', 'strength': 0.7, 'nsfw_only': False},
            {'file': 'klein/hot.safetensors', 'strength': 1.0, 'nsfw_only': True},
        ]
        assert 'ultra_real_lora' not in conf['klein']
        assert 'nsfw_lora' not in conf['klein']
        assert 'ultra_real_strength' not in conf['klein']
        assert 'nsfw_strength' not in conf['klein']
        # Idempotent: loading again does not duplicate the entries.
        again = cfg.load_config(force=True)
        assert again['klein']['generation_loras'] == conf['klein']['generation_loras']


def test_migration_skips_empty_keys_and_existing_entries(app, tmp_path):
    from app import config as cfg
    with app.app_context():
        p = os.environ['LDS_CONFIG']
        with open(p, 'w', encoding='utf-8') as fh:
            json.dump({'klein': {
                'ultra_real_lora': '',                   # empty -> no entry
                'nsfw_lora': 'klein/hot.safetensors',
                'generation_loras': [                    # already migrated once
                    {'file': 'klein/hot.safetensors', 'strength': 0.9, 'nsfw_only': True}],
            }}, fh)
        conf = cfg.load_config(force=True)
        assert conf['klein']['generation_loras'] == [
            {'file': 'klein/hot.safetensors', 'strength': 0.9, 'nsfw_only': True}]


def test_save_purges_legacy_keys_from_the_file(app):
    """Deleting a migrated row must STICK: the first save rewrites the file
    without the legacy keys, so they can't resurrect the row at the next load."""
    from app import config as cfg
    with app.app_context():
        p = os.environ['LDS_CONFIG']
        with open(p, 'w', encoding='utf-8') as fh:
            json.dump({'klein': {'ultra_real_lora': 'klein/tex.safetensors'}}, fh)
        cfg.load_config(force=True)          # migration visible in memory
        # User deletes the migrated row in Settings -> PUT saves an empty list.
        cfg.save_config({'klein': {'generation_loras': []}})
        on_disk = json.load(open(p, encoding='utf-8'))
        assert 'ultra_real_lora' not in on_disk['klein']
        assert on_disk['klein']['generation_loras'] == []
        assert cfg.get('klein.generation_loras') == []   # row did NOT resurrect


# --- Helpers -----------------------------------------------------------------
def test_configured_list_sanitized_ordered_capped(app):
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    with app.app_context():
        many = [{'file': f'klein/l{i}.safetensors', 'strength': 0.5} for i in range(12)]
        cfg.save_config({'klein': {'generation_loras': [
            {'file': '  '}, 'junk', {'strength': 1},          # dropped
            {'file': 'klein/a.safetensors', 'strength': 'x'},  # junk strength -> 0.6
            {'file': 'klein/b.safetensors', 'strength': 9, 'nsfw_only': 1},
        ] + many}})
        out = keh.configured_generation_loras()
        assert len(out) == keh.MAX_GENERATION_LORAS
        assert out[0] == {'file': 'klein/a.safetensors', 'strength': 0.6, 'nsfw_only': False}
        assert out[1] == {'file': 'klein/b.safetensors', 'strength': 1.5, 'nsfw_only': True}
        assert out[2]['file'] == 'klein/l0.safetensors'


def test_resolve_run_keeps_config_order_and_nsfw_flag(app):
    """The request can neither reorder the chain, nor arm an unknown file, nor
    strip a row's nsfw_only flag — only pick rows and set their strengths."""
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    with app.app_context():
        cfg.save_config({'klein': {'generation_loras': GEN_LORAS}})
        out = keh.resolve_run_generation_loras([
            {'file': 'klein/gen-c.safetensors', 'strength': 1.2},   # reversed order
            {'file': 'klein/gen-b-hot.safetensors', 'strength': 0.3, 'nsfw_only': False},
            {'file': 'klein/evil-unconfigured.safetensors', 'strength': 1.0},
        ])
        assert [e['file'] for e in out] == \
            ['klein/gen-b-hot.safetensors', 'klein/gen-c.safetensors']  # config order
        assert out[0]['nsfw_only'] is True     # flag comes from CONFIG, not request
        assert out[0]['strength'] == 0.3
        assert out[1]['strength'] == 1.2
        assert keh.resolve_run_generation_loras(None) == []
        assert keh.resolve_run_generation_loras('junk') == []


# --- Graph wiring ------------------------------------------------------------
def test_chain_of_three_in_list_order(app, tmp_path, monkeypatch):
    """N=3: 114 -> consistency -> gen_1 -> gen_2 -> gen_3 -> 139 (base LoRA
    present), each link a LoraLoaderModelOnly hanging off the previous one."""
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    from app.job_queue import queue_manager
    with app.app_context():
        _comfy(tmp_path, cfg, base_lora=True)
        wf = _enqueue(keh, queue_manager, monkeypatch, tmp_path,
                      generation_loras=[dict(e) for e in GEN_LORAS])
        assert wf['ds_consistency_lora']['inputs']['model'] == ['114', 0]
        assert wf['ds_gen_lora_1']['inputs']['model'] == ['ds_consistency_lora', 0]
        assert wf['ds_gen_lora_2']['inputs']['model'] == ['ds_gen_lora_1', 0]
        assert wf['ds_gen_lora_3']['inputs']['model'] == ['ds_gen_lora_2', 0]
        assert wf['139']['inputs']['model'] == ['ds_gen_lora_3', 0]
        assert wf['ds_gen_lora_1']['inputs']['lora_name'] == \
            os.path.join('klein', 'gen-a.safetensors')
        assert wf['ds_gen_lora_2']['inputs']['lora_name'] == \
            os.path.join('klein', 'gen-b-hot.safetensors')
        assert wf['ds_gen_lora_3']['inputs']['lora_name'] == \
            os.path.join('klein', 'gen-c.safetensors')
        assert wf['ds_gen_lora_1']['inputs']['strength_model'] == 0.6
        assert all(wf[n]['class_type'] == 'LoraLoaderModelOnly'
                   for n in ('ds_gen_lora_1', 'ds_gen_lora_2', 'ds_gen_lora_3'))


def test_chain_survives_base_lora_bypass(app, tmp_path, monkeypatch):
    """Base 'realistic' LoRA absent -> node 139 bypassed -> 102 wires straight
    to the LAST generation LoRA (the chain stays intact)."""
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    from app.job_queue import queue_manager
    with app.app_context():
        _comfy(tmp_path, cfg)              # no realistic.safetensors on disk
        wf = _enqueue(keh, queue_manager, monkeypatch, tmp_path,
                      generation_loras=[dict(e) for e in GEN_LORAS])
        assert '139' not in wf
        assert wf['102']['inputs']['model'] == ['ds_gen_lora_3', 0]


def test_chain_hangs_off_unet_when_consistency_is_off(app, tmp_path, monkeypatch):
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    from app.job_queue import queue_manager
    with app.app_context():
        _comfy(tmp_path, cfg)
        wf = _enqueue(keh, queue_manager, monkeypatch, tmp_path, lora_strength=0,
                      generation_loras=[dict(GEN_LORAS[0])])
        assert 'ds_consistency_lora' not in wf
        assert wf['ds_gen_lora_1']['inputs']['model'] == ['114', 0]
        assert wf['102']['inputs']['model'] == ['ds_gen_lora_1', 0]


def test_enqueue_caps_the_chain(app, tmp_path, monkeypatch):
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    from app.job_queue import queue_manager
    with app.app_context():
        stems = [f'many-{i}' for i in range(keh.MAX_GENERATION_LORAS + 3)]
        _comfy(tmp_path, cfg, gen_files=stems, config_loras=[])
        wf = _enqueue(keh, queue_manager, monkeypatch, tmp_path,
                      generation_loras=[{'file': f'klein/{s}.safetensors', 'strength': 0.5}
                                        for s in stems])
        injected = [k for k in wf if k.startswith('ds_gen_lora_')]
        assert len(injected) == keh.MAX_GENERATION_LORAS


# --- Silent degradation ------------------------------------------------------
def test_no_rows_means_no_extra_nodes(app, tmp_path, monkeypatch):
    """Files configured AND on disk, but nothing armed for this run -> nothing
    injected (a configured row alone never arms itself)."""
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    from app.job_queue import queue_manager
    with app.app_context():
        _comfy(tmp_path, cfg)
        wf = _enqueue(keh, queue_manager, monkeypatch, tmp_path)
        assert not any(k.startswith('ds_gen_lora_') for k in wf)
        assert wf['102']['inputs']['model'] == ['ds_consistency_lora', 0]


def test_missing_middle_file_degrades_that_row_only(app, tmp_path, monkeypatch):
    """The middle row's file is NOT on disk -> that row is skipped with a log,
    the surrounding rows still chain up (per-row degradation)."""
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    from app.job_queue import queue_manager
    with app.app_context():
        _comfy(tmp_path, cfg, gen_files=('gen-a', 'gen-c'))   # gen-b-hot absent
        wf = _enqueue(keh, queue_manager, monkeypatch, tmp_path,
                      generation_loras=[dict(e) for e in GEN_LORAS])
        assert 'ds_gen_lora_2' not in wf                       # degraded row
        assert wf['ds_gen_lora_1']['inputs']['model'] == ['ds_consistency_lora', 0]
        assert wf['ds_gen_lora_3']['inputs']['model'] == ['ds_gen_lora_1', 0]
        assert wf['102']['inputs']['model'] == ['ds_gen_lora_3', 0]


def test_zero_strength_row_is_skipped(app, tmp_path, monkeypatch):
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    from app.job_queue import queue_manager
    with app.app_context():
        _comfy(tmp_path, cfg)
        wf = _enqueue(keh, queue_manager, monkeypatch, tmp_path,
                      generation_loras=[{'file': 'klein/gen-a.safetensors', 'strength': 0},
                                        {'file': 'klein/gen-c.safetensors', 'strength': 0.5}])
        assert 'ds_gen_lora_1' not in wf
        assert wf['ds_gen_lora_2']['inputs']['model'] == ['ds_consistency_lora', 0]
        assert wf['102']['inputs']['model'] == ['ds_gen_lora_2', 0]


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


def test_fanout_gates_nsfw_only_rows_per_variation(app, tmp_path, monkeypatch):
    """Mixed batch with all three rows armed: nsfw_only rows ride ONLY the NSFW
    variations; plain rows ride every one, order preserved."""
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
        ], 1, None, generation_loras=ARM_ALL)
        wfs = [c['workflow_data'] for c in captured]

        def chained(wf):
            return [wf[k]['inputs']['lora_name'] for k in
                    sorted((k for k in wf if k.startswith('ds_gen_lora_')),
                           key=lambda k: int(k.rsplit('_', 1)[1]))]
        hot = os.path.join('klein', 'gen-b-hot.safetensors')
        a, c = os.path.join('klein', 'gen-a.safetensors'), os.path.join('klein', 'gen-c.safetensors')
        assert chained(wfs[0]) == [a, hot, c]     # NSFW catalog label -> full chain
        assert chained(wfs[1]) == [a, hot, c]     # explicit nsfw flag (🔞 custom)
        assert chained(wfs[2]) == [a, c]          # SFW shot: nsfw_only row dropped


def test_generate_route_passes_generation_loras(app, client, tmp_path, monkeypatch):
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
        'generation_loras': ARM_ALL})
    assert resp.status_code == 200
    wf = captured[0]['workflow_data']
    assert 'ds_gen_lora_1' in wf and 'ds_gen_lora_3' in wf


def test_generate_route_without_the_key_keeps_rows_off(app, client, tmp_path, monkeypatch):
    """Legacy/omitted body key (the default UI state) -> no generation-LoRA
    nodes even with the whole list configured and on disk."""
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
    assert not any(k.startswith('ds_gen_lora_') for k in captured[0]['workflow_data'])


def test_regenerate_gates_nsfw_only_on_the_tiles_label(app, tmp_path, monkeypatch):
    """Regenerate: an SFW tile never receives an nsfw_only row even when the
    caller arms it; an NSFW-labelled tile receives the full chain."""
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
                             generation_loras=ARM_ALL)
        svc.regenerate_image(LOCAL_USER, rows['Corps, nu debout'],
                             generation_loras=ARM_ALL)
        sfw_wf, nsfw_wf = captured[0]['workflow_data'], captured[1]['workflow_data']
        sfw_names = [sfw_wf[k]['inputs']['lora_name'] for k in sfw_wf
                     if k.startswith('ds_gen_lora_')]
        assert os.path.join('klein', 'gen-b-hot.safetensors') not in sfw_names
        assert len(sfw_names) == 2
        nsfw_names = [nsfw_wf[k]['inputs']['lora_name'] for k in nsfw_wf
                      if k.startswith('ds_gen_lora_')]
        assert os.path.join('klein', 'gen-b-hot.safetensors') in nsfw_names
        assert len(nsfw_names) == 3


# --- Settings round-trip -----------------------------------------------------
def test_config_roundtrip_through_settings_api(app, client):
    resp = client.put('/api/settings', json={'config': {'klein': {
        'generation_loras': GEN_LORAS}}})
    assert resp.status_code == 200
    saved = client.get('/api/settings').get_json()['config']['klein']
    assert saved['generation_loras'] == GEN_LORAS
    # The untouched neighbours survive the partial save.
    assert saved['consistency_strength'] == 0.5
