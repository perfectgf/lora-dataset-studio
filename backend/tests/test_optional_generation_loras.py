"""Optional generation-LoRA PRESETS (Idea by @waltm — Discord feature request).

The user defines named combinations (`klein.generation_lora_presets`, entries
{name, loras: [{file, strength}]}); per run only a preset NAME is sent and the
backend resolves the chain from CONFIG (fail-closed). These tests pin:
(a) the graph wiring 114 -> consistency -> gen_1 -> ... -> gen_N -> 139 for a
    preset with N > 2, order preserved, caps (8 rows/preset, 12 presets);
(b) the two-stage soft migration: very old single-slot keys -> flat list ->
    ONE 'My LoRAs' preset; idempotent, legacy keys dropped, purged from the
    file on save, deleted preset never resurrects;
(c) per-row silent degradation (missing file / blank / strength 0) with the
    rest of the chain still linking up;
(d) preset selection semantics: the SAME chain applies to EVERY variation of
    the run (SFW and NSFW alike — the old per-variation badge gating is GONE),
    and an unknown preset name is ignored cleanly (no extra nodes, run works);
(e) requests can only NAME a configured preset (never define files/order) and
    the config round-trip via /api/settings.
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


# Ordered stand-in rows of the test preset.
GEN_ROWS = [
    {'file': 'klein/gen-a.safetensors', 'strength': 0.6},
    {'file': 'klein/gen-b.safetensors', 'strength': 0.8},
    {'file': 'klein/gen-c.safetensors', 'strength': 0.5},
]
PRESETS = [{'name': 'Full stack', 'loras': GEN_ROWS},
           {'name': 'Just one', 'loras': [GEN_ROWS[0]]}]


def _comfy(tmp_path, cfg, gen_files=('gen-a', 'gen-b', 'gen-c'),
           base_lora=False, presets=PRESETS):
    """A ComfyUI tree with all REQUIRED Klein assets + the consistency LoRA, a
    configurable subset of the generation-LoRA FILES on disk, and the preset
    list configured (files on disk or not — degradation is the point)."""
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
                     'klein': {'generation_lora_presets': presets}})
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


def _gen_chain(wf):
    """Ordered lora_name list of the injected ds_gen_lora_* nodes."""
    keys = sorted((k for k in wf if k.startswith('ds_gen_lora_')),
                  key=lambda k: int(k.rsplit('_', 1)[1]))
    return [wf[k]['inputs']['lora_name'] for k in keys]


# --- Defaults & migration ----------------------------------------------------
def test_config_defaults_empty_presets_no_legacy_keys(app):
    from app import config as cfg
    with app.app_context():
        assert cfg.get('klein.generation_lora_presets') == []
        assert cfg.get('klein.generation_loras') is None
        assert cfg.get('klein.ultra_real_lora') is None
        assert cfg.get('klein.nsfw_lora') is None


def test_flat_list_migrates_into_a_named_preset(app):
    """Migration (a): the intermediate flat generation_loras list becomes ONE
    'My LoRAs' preset (order kept, nsfw_only flags dropped — presets carry the
    intent now), the flat key is removed, and reloading never duplicates."""
    from app import config as cfg
    with app.app_context():
        p = os.environ['LDS_CONFIG']
        with open(p, 'w', encoding='utf-8') as fh:
            json.dump({'klein': {'generation_loras': [
                {'file': 'klein/tex.safetensors', 'strength': 0.7, 'nsfw_only': False},
                {'file': 'klein/hot.safetensors', 'strength': 1.0, 'nsfw_only': True},
            ]}}, fh)
        conf = cfg.load_config(force=True)
        assert conf['klein']['generation_lora_presets'] == [
            {'name': 'My LoRAs', 'loras': [
                {'file': 'klein/tex.safetensors', 'strength': 0.7},
                {'file': 'klein/hot.safetensors', 'strength': 1.0}]}]
        assert 'generation_loras' not in conf['klein']
        # Idempotent: loading again does not duplicate the preset.
        again = cfg.load_config(force=True)
        assert again['klein']['generation_lora_presets'] == conf['klein']['generation_lora_presets']


def test_two_slot_keys_migrate_through_to_the_preset(app):
    """Migration (b): the VERY old single-slot ultra_real/nsfw keys chain
    through both stages into the same 'My LoRAs' preset, strengths carried,
    every legacy key dropped."""
    from app import config as cfg
    with app.app_context():
        p = os.environ['LDS_CONFIG']
        with open(p, 'w', encoding='utf-8') as fh:
            json.dump({'klein': {'ultra_real_lora': 'klein/tex.safetensors',
                                 'ultra_real_strength': 0.7,
                                 'nsfw_lora': 'klein/hot.safetensors',
                                 'nsfw_strength': 1.0}}, fh)
        conf = cfg.load_config(force=True)
        assert conf['klein']['generation_lora_presets'] == [
            {'name': 'My LoRAs', 'loras': [
                {'file': 'klein/tex.safetensors', 'strength': 0.7},
                {'file': 'klein/hot.safetensors', 'strength': 1.0}]}]
        for legacy in ('ultra_real_lora', 'ultra_real_strength',
                       'nsfw_lora', 'nsfw_strength', 'generation_loras'):
            assert legacy not in conf['klein']


def test_migration_skips_when_the_preset_name_already_exists(app):
    from app import config as cfg
    with app.app_context():
        p = os.environ['LDS_CONFIG']
        with open(p, 'w', encoding='utf-8') as fh:
            json.dump({'klein': {
                'generation_loras': [{'file': 'klein/tex.safetensors', 'strength': 0.7}],
                'generation_lora_presets': [{'name': 'My LoRAs', 'loras': []}],
            }}, fh)
        conf = cfg.load_config(force=True)
        # Existing 'My LoRAs' preset wins — no duplicate, flat list dropped.
        assert conf['klein']['generation_lora_presets'] == [{'name': 'My LoRAs', 'loras': []}]


def test_save_purges_legacy_keys_and_deleted_preset_stays_deleted(app):
    """A save that explicitly carries the presets does NOT reconvert the
    legacy keys — deleting the migrated preset sticks, and the file is
    purged of every legacy key."""
    from app import config as cfg
    with app.app_context():
        p = os.environ['LDS_CONFIG']
        with open(p, 'w', encoding='utf-8') as fh:
            json.dump({'klein': {'ultra_real_lora': 'klein/tex.safetensors',
                                 'generation_loras': [{'file': 'klein/x.safetensors'}]}}, fh)
        cfg.load_config(force=True)          # migration visible in memory
        # User deletes the migrated preset in Settings -> PUT saves an empty list.
        cfg.save_config({'klein': {'generation_lora_presets': []}})
        on_disk = json.load(open(p, encoding='utf-8'))
        assert 'ultra_real_lora' not in on_disk['klein']
        assert 'generation_loras' not in on_disk['klein']
        assert on_disk['klein']['generation_lora_presets'] == []
        assert cfg.get('klein.generation_lora_presets') == []   # no resurrection


# --- Helpers -----------------------------------------------------------------
def test_configured_presets_sanitized_ordered_capped(app):
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    with app.app_context():
        many_rows = [{'file': f'klein/l{i}.safetensors', 'strength': 0.5} for i in range(12)]
        many_presets = [{'name': f'P{i}', 'loras': []} for i in range(15)]
        cfg.save_config({'klein': {'generation_lora_presets': [
            {'name': '  ', 'loras': GEN_ROWS},          # blank name -> dropped
            'junk', {'loras': GEN_ROWS},                # malformed -> dropped
            {'name': 'Big', 'loras': [
                {'file': '  '}, 'junk',                 # bad rows -> dropped
                {'file': 'klein/a.safetensors', 'strength': 'x'},   # junk -> 0.6
                {'file': 'klein/b.safetensors', 'strength': 9},     # clamp 1.5
            ] + many_rows},
            {'name': 'Big', 'loras': []},               # duplicate name -> dropped
        ] + many_presets}})
        out = keh.configured_generation_lora_presets()
        assert len(out) == keh.MAX_GENERATION_LORA_PRESETS
        big = out[0]
        assert big['name'] == 'Big'
        assert len(big['loras']) == keh.MAX_GENERATION_LORAS
        assert big['loras'][0] == {'file': 'klein/a.safetensors', 'strength': 0.6}
        assert big['loras'][1] == {'file': 'klein/b.safetensors', 'strength': 1.5}
        assert out[1]['name'] == 'P0'


def test_resolve_preset_by_name_fail_closed(app):
    """The request can only NAME a preset: known -> its ordered rows; unknown /
    blank / None -> [] (degrade to no extra LoRAs, never an error)."""
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    with app.app_context():
        cfg.save_config({'klein': {'generation_lora_presets': PRESETS}})
        rows = keh.resolve_generation_lora_preset('Full stack')
        assert [r['file'] for r in rows] == [e['file'] for e in GEN_ROWS]
        assert keh.resolve_generation_lora_preset('Just one') == [GEN_ROWS[0]]
        assert keh.resolve_generation_lora_preset('No such preset') == []
        assert keh.resolve_generation_lora_preset('') == []
        assert keh.resolve_generation_lora_preset(None) == []
        assert keh.resolve_generation_lora_preset(42) == []


# --- Graph wiring ------------------------------------------------------------
def test_chain_of_three_in_preset_order(app, tmp_path, monkeypatch):
    """N=3: 114 -> consistency -> gen_1 -> gen_2 -> gen_3 -> 139 (base LoRA
    present), each link a LoraLoaderModelOnly hanging off the previous one."""
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    from app.job_queue import queue_manager
    with app.app_context():
        _comfy(tmp_path, cfg, base_lora=True)
        wf = _enqueue(keh, queue_manager, monkeypatch, tmp_path,
                      generation_loras=keh.resolve_generation_lora_preset('Full stack'))
        assert wf['ds_consistency_lora']['inputs']['model'] == ['114', 0]
        assert wf['ds_gen_lora_1']['inputs']['model'] == ['ds_consistency_lora', 0]
        assert wf['ds_gen_lora_2']['inputs']['model'] == ['ds_gen_lora_1', 0]
        assert wf['ds_gen_lora_3']['inputs']['model'] == ['ds_gen_lora_2', 0]
        assert wf['139']['inputs']['model'] == ['ds_gen_lora_3', 0]
        assert _gen_chain(wf) == [os.path.join('klein', 'gen-a.safetensors'),
                                  os.path.join('klein', 'gen-b.safetensors'),
                                  os.path.join('klein', 'gen-c.safetensors')]
        assert wf['ds_gen_lora_1']['inputs']['strength_model'] == 0.6
        assert all(wf[f'ds_gen_lora_{i}']['class_type'] == 'LoraLoaderModelOnly'
                   for i in (1, 2, 3))


def test_chain_survives_base_lora_bypass(app, tmp_path, monkeypatch):
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    from app.job_queue import queue_manager
    with app.app_context():
        _comfy(tmp_path, cfg)              # no realistic.safetensors on disk
        wf = _enqueue(keh, queue_manager, monkeypatch, tmp_path,
                      generation_loras=keh.resolve_generation_lora_preset('Full stack'))
        assert '139' not in wf
        assert wf['102']['inputs']['model'] == ['ds_gen_lora_3', 0]


def test_chain_hangs_off_unet_when_consistency_is_off(app, tmp_path, monkeypatch):
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    from app.job_queue import queue_manager
    with app.app_context():
        _comfy(tmp_path, cfg)
        wf = _enqueue(keh, queue_manager, monkeypatch, tmp_path, lora_strength=0,
                      generation_loras=keh.resolve_generation_lora_preset('Just one'))
        assert 'ds_consistency_lora' not in wf
        assert wf['ds_gen_lora_1']['inputs']['model'] == ['114', 0]
        assert wf['102']['inputs']['model'] == ['ds_gen_lora_1', 0]


def test_enqueue_caps_the_chain(app, tmp_path, monkeypatch):
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    from app.job_queue import queue_manager
    with app.app_context():
        stems = [f'many-{i}' for i in range(keh.MAX_GENERATION_LORAS + 3)]
        _comfy(tmp_path, cfg, gen_files=stems, presets=[])
        wf = _enqueue(keh, queue_manager, monkeypatch, tmp_path,
                      generation_loras=[{'file': f'klein/{s}.safetensors', 'strength': 0.5}
                                        for s in stems])
        assert len(_gen_chain(wf)) == keh.MAX_GENERATION_LORAS


# --- Silent degradation ------------------------------------------------------
def test_no_preset_means_no_extra_nodes(app, tmp_path, monkeypatch):
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    from app.job_queue import queue_manager
    with app.app_context():
        _comfy(tmp_path, cfg)
        wf = _enqueue(keh, queue_manager, monkeypatch, tmp_path)
        assert not any(k.startswith('ds_gen_lora_') for k in wf)
        assert wf['102']['inputs']['model'] == ['ds_consistency_lora', 0]


def test_missing_middle_file_degrades_that_row_only(app, tmp_path, monkeypatch):
    """The preset's middle file is NOT on disk -> that row is skipped with a
    log, the surrounding rows still chain up (per-row degradation)."""
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    from app.job_queue import queue_manager
    with app.app_context():
        _comfy(tmp_path, cfg, gen_files=('gen-a', 'gen-c'))   # gen-b absent
        wf = _enqueue(keh, queue_manager, monkeypatch, tmp_path,
                      generation_loras=keh.resolve_generation_lora_preset('Full stack'))
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


# --- Preset selection at the service/route level -----------------------------
def _dataset_with_ref(svc, LOCAL_USER):
    ds = svc.create_dataset(LOCAL_USER, 'Loras', 'loras')
    d = svc._dataset_dir(ds.id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'ref.webp'), 'wb') as fh:
        fh.write(_png())
    ds.ref_filename = 'ref.webp'
    svc.db.session.commit()
    return ds


def test_fanout_applies_the_preset_to_every_variation(app, tmp_path, monkeypatch):
    """The preset's chain rides EVERY variation of the run — SFW and NSFW
    alike. Pins that the old per-variation 🔞 badge gating is GONE: the chosen
    preset carries the intent."""
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
            {'label': 'Corps debout face', 'framing': 'body', 'prompt': 'standing'},
        ], 1, None, generation_lora_preset='Full stack')
        chains = [_gen_chain(c['workflow_data']) for c in captured]
        expected = [os.path.join('klein', 'gen-a.safetensors'),
                    os.path.join('klein', 'gen-b.safetensors'),
                    os.path.join('klein', 'gen-c.safetensors')]
        assert chains == [expected, expected]      # identical chain, NSFW and SFW


def test_generate_route_passes_the_preset_name(app, client, tmp_path, monkeypatch):
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
        'variations': [{'label': 'x', 'framing': 'face', 'prompt': 'p'}],
        'generation_lora_preset': 'Full stack'})
    assert resp.status_code == 200
    assert len(_gen_chain(captured[0]['workflow_data'])) == 3


def test_generate_route_unknown_preset_is_ignored_cleanly(app, client, tmp_path, monkeypatch):
    """A stale/renamed preset name must degrade to 'no extra LoRAs' (logged),
    never fail the run or invent a chain."""
    from app import config as cfg
    from app.job_queue import queue_manager
    captured = []
    monkeypatch.setattr(queue_manager, 'add_job',
                        lambda **kw: (captured.append(kw), kw['job_id'])[1])
    with app.app_context():
        _comfy(tmp_path, cfg)
    ds_id = client.post('/api/dataset/create',
                        json={'name': 'U', 'trigger_word': 'u'}).get_json()['id']
    client.post(f'/api/dataset/{ds_id}/ref',
                data={'file': (io.BytesIO(_png()), 'ref.png')},
                content_type='multipart/form-data')
    resp = client.post(f'/api/dataset/{ds_id}/generate', json={
        'generator': 'klein', 'multiplier': 1,
        'variations': [{'label': 'x', 'framing': 'face', 'prompt': 'p'}],
        'generation_lora_preset': 'Renamed away'})
    assert resp.status_code == 200
    assert resp.get_json()['created'] == 1
    assert _gen_chain(captured[0]['workflow_data']) == []


def test_generate_route_without_the_key_keeps_presets_off(app, client, tmp_path, monkeypatch):
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
    assert _gen_chain(captured[0]['workflow_data']) == []


def test_regenerate_applies_the_preset_regardless_of_label(app, tmp_path, monkeypatch):
    """Regenerate: the preset rides on any tile — SFW-labelled included (no
    badge gating), resolved from config by name only."""
    from app import config as cfg
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app.job_queue import queue_manager
    with app.app_context():
        _comfy(tmp_path, cfg)
        ds = _dataset_with_ref(svc, LOCAL_USER)
        img = svc.FaceDatasetImage(dataset_id=ds.id, source='generated',
                                   status='finished', variation_label='Corps debout face',
                                   variation_prompt='p', framing='body')
        svc.db.session.add(img)
        svc.db.session.commit()
        captured = []
        monkeypatch.setattr(queue_manager, 'add_job',
                            lambda **kw: (captured.append(kw), kw['job_id'])[1])
        svc.regenerate_image(LOCAL_USER, img.id, generation_lora_preset='Just one')
        assert _gen_chain(captured[0]['workflow_data']) == \
            [os.path.join('klein', 'gen-a.safetensors')]
        # Unknown preset degrades to none, run still succeeds.
        captured.clear()
        svc.regenerate_image(LOCAL_USER, img.id, generation_lora_preset='ghost')
        assert _gen_chain(captured[0]['workflow_data']) == []


# --- Settings round-trip -----------------------------------------------------
def test_config_roundtrip_through_settings_api(app, client):
    resp = client.put('/api/settings', json={'config': {'klein': {
        'generation_lora_presets': PRESETS}}})
    assert resp.status_code == 200
    saved = client.get('/api/settings').get_json()['config']['klein']
    assert saved['generation_lora_presets'] == PRESETS
    # The untouched neighbours survive the partial save.
    assert saved['consistency_strength'] == 0.5
