"""Always-on generation LoRAs on the Krea 2 Identity Edit dataset lane.

The mechanism mirrors the Klein one (Idea by @waltm) with a different clamp: the
utility LoRAs this exists for have no effect below ~10, so the ceiling is 20 —
comfyui.inject_krea_loras' clamp — not Klein's 1.5.

Nothing here renders anything: no GPU second, no paid call.
"""
import json
import os

import pytest
from PIL import Image


PRESETS = [
    {'name': 'Bypass', 'loras': [
        {'file': 'krea/krea2filterbypass3.safetensors', 'strength': 13.0},
        {'file': 'krea/detail_slider.safetensors', 'strength': 0.6},
    ]},
    {'name': 'Just detail', 'loras': [
        {'file': 'krea/detail_slider.safetensors', 'strength': 0.6},
    ]},
]


def _set_presets(presets):
    from app import config as cfg
    cfg.save_config({'krea': {'generation_lora_presets': presets}})


def _krea_dataset(app, svc):
    """A dataset with a reference image, ready for the Krea fan-out. Preflight is
    bypassed by the tests that monkeypatch enqueue_krea_edit — they assert wiring,
    not installation. `trigger_word` is NOT NULL in the model, so it is set here:
    omitting it fails with an IntegrityError that looks unrelated to this feature."""
    from app.models import FaceDataset, db
    ds = FaceDataset(user_id='local', name='ds', trigger_word='zzt',
                     ref_filename='ref.png', train_type='krea')
    db.session.add(ds)
    db.session.commit()
    path = svc._dataset_path(ds.id)
    os.makedirs(path, exist_ok=True)
    Image.new('RGB', (512, 512), (10, 20, 30)).save(os.path.join(path, 'ref.png'))
    return ds


# --- Defaults ----------------------------------------------------------------

def test_default_is_an_empty_preset_list(app):
    from app import config as cfg
    assert cfg.get('krea.generation_lora_presets') == []


def test_presets_survive_a_save_of_an_unrelated_section(app):
    """save_config deep-merges over the RAW file, so a list absent from the
    partial is preserved. This is why the Krea lane needs no carve-out."""
    from app import config as cfg
    _set_presets(PRESETS)
    cfg.save_config({'krea': {'steps': 12}})
    stored = cfg.get('krea.generation_lora_presets')
    assert [p['name'] for p in stored] == ['Bypass', 'Just detail']
    assert cfg.get('krea.steps') == 12


# --- Sanitizer ---------------------------------------------------------------

def test_configured_presets_sanitized_ordered_capped(app):
    from app.services import krea_edit_helper as keh
    _set_presets([
        {'name': '  Bypass  ', 'loras': [
            {'file': ' krea/a.safetensors ', 'strength': 13},
            {'file': '', 'strength': 1},                     # blank file -> dropped
            {'file': 'krea/b.safetensors', 'strength': 'x'},  # junk -> default
            {'file': 'krea/c.safetensors', 'strength': -4},   # negative -> 0
            {'file': 'krea/d.safetensors', 'strength': 999},  # over -> 20
        ]},
        {'name': 'Bypass', 'loras': []},   # duplicate name -> dropped
        {'name': '', 'loras': []},          # blank name -> dropped
        'not a dict',
    ])
    presets = keh.configured_generation_lora_presets()
    assert [p['name'] for p in presets] == ['Bypass']
    assert presets[0]['loras'] == [
        {'file': 'krea/a.safetensors', 'strength': 13.0},
        {'file': 'krea/b.safetensors', 'strength': keh.DEFAULT_ROW_STRENGTH},
        {'file': 'krea/c.safetensors', 'strength': 0.0},
        {'file': 'krea/d.safetensors', 'strength': keh.LORA_STRENGTH_MAX},
    ]


def test_row_and_preset_caps(app):
    from app.services import krea_edit_helper as keh
    rows = [{'file': f'krea/{i}.safetensors', 'strength': 1.0} for i in range(20)]
    _set_presets([{'name': f'P{i}', 'loras': rows} for i in range(20)])
    presets = keh.configured_generation_lora_presets()
    assert len(presets) == keh.MAX_GENERATION_LORA_PRESETS
    assert len(presets[0]['loras']) == keh.MAX_GENERATION_LORAS


# --- Resolution (fail-closed) ------------------------------------------------

def test_resolve_by_name(app):
    from app.services import krea_edit_helper as keh
    _set_presets(PRESETS)
    assert keh.resolve_generation_lora_preset('Bypass') == [
        {'file': 'krea/krea2filterbypass3.safetensors', 'strength': 13.0},
        {'file': 'krea/detail_slider.safetensors', 'strength': 0.6},
    ]


@pytest.mark.parametrize('name', ['', None, '   ', 'Renamed since', 42])
def test_resolve_fail_closed(app, name):
    """Blank, unknown and non-string names all mean 'no extra LoRAs' — never an
    exception, so a stale UI can't stop someone generating."""
    from app.services import krea_edit_helper as keh
    _set_presets(PRESETS)
    assert keh.resolve_generation_lora_preset(name) == []


def test_resolved_rows_are_copies(app):
    """A caller mutating its rows must not corrupt the next run's config view."""
    from app.services import krea_edit_helper as keh
    _set_presets(PRESETS)
    rows = keh.resolve_generation_lora_preset('Bypass')
    rows[0]['strength'] = 0.0
    assert keh.resolve_generation_lora_preset('Bypass')[0]['strength'] == 13.0


# --- Graph wiring ------------------------------------------------------------

def _graph(**kw):
    from app.services import krea_edit_helper as keh
    return keh.build_workflow('ref.png', 'a prompt', unet='Krea/base.safetensors',
                              clip='te.safetensors', vae='vae.safetensors',
                              lora_name='krea/id.safetensors', width=1024,
                              height=1024, seed=7, **kw)


ROWS = [{'file': 'krea/bypass.safetensors', 'strength': 13.0},
        {'file': 'krea/detail.safetensors', 'strength': 0.6}]


def test_no_rows_leaves_the_graph_exactly_as_it_was():
    """The regression guard for every existing Krea run: a user with no presets
    must get byte-identical output to before this feature."""
    assert _graph(generation_loras=None) == _graph()
    assert _graph(generation_loras=[]) == _graph()
    assert not any(k.startswith('gen_lora_') for k in _graph())


def test_rows_chain_in_order_between_the_identity_lora_and_the_patch():
    g = _graph(generation_loras=ROWS)
    assert g['gen_lora_1']['inputs']['model'] == ['4', 0]
    assert g['gen_lora_1']['inputs']['lora_name'] == 'krea/bypass.safetensors'
    assert g['gen_lora_1']['inputs']['strength_model'] == 13.0
    assert g['gen_lora_2']['inputs']['model'] == ['gen_lora_1', 0]
    assert g['gen_lora_2']['inputs']['lora_name'] == 'krea/detail.safetensors'
    # The patch — and through it the KSampler — sees the END of the chain.
    assert g['7']['inputs']['model'] == ['gen_lora_2', 0]
    assert g['11']['inputs']['model'] == ['7', 0]
    assert all(g[k]['class_type'] == 'LoraLoaderModelOnly'
               for k in ('4', 'gen_lora_1', 'gen_lora_2'))


def test_the_identity_lora_still_hangs_off_the_unet_with_rows_present():
    """The stack goes AFTER the identity LoRA — it never displaces it."""
    g = _graph(generation_loras=ROWS)
    assert g['4']['inputs']['model'] == ['1', 0]
    assert g['4']['inputs']['lora_name'] == 'krea/id.safetensors'


def test_graph_injection_caps_the_chain():
    from app.services import krea_edit_helper as keh
    rows = [{'file': f'krea/{i}.safetensors', 'strength': 1.0} for i in range(20)]
    g = _graph(generation_loras=rows)
    chained = [k for k in g if k.startswith('gen_lora_')]
    assert len(chained) == keh.MAX_GENERATION_LORAS
    assert g['7']['inputs']['model'] == [f'gen_lora_{keh.MAX_GENERATION_LORAS}', 0]


def test_build_workflow_stays_pure(monkeypatch):
    """Its docstring promises no config read and no disk access — the existence
    checks live in enqueue_krea_edit. Guard it, because a future 'just check the
    file here' is exactly what would break every graph test."""
    from app import config as cfg
    from app.services import krea_edit_helper as keh
    monkeypatch.setattr(cfg, 'get', lambda *a, **k: pytest.fail('config read'))
    monkeypatch.setattr(keh.os.path, 'exists', lambda *a: pytest.fail('disk access'))
    assert _graph(generation_loras=ROWS)['7']['inputs']['model'] == ['gen_lora_2', 0]


# --- Row survival at enqueue time -------------------------------------------

def _comfy_with_loras(tmp_path, present=('bypass',)):
    """A ComfyUI tree whose loras root holds only `present`, so the rest of a
    preset's rows are legitimately missing."""
    base = tmp_path / 'comfyui'
    for sub in ('input', 'output'):
        (base / sub).mkdir(parents=True, exist_ok=True)
    (base / 'main.py').write_text('# fake', encoding='utf-8')
    loras = base / 'models' / 'loras' / 'krea'
    loras.mkdir(parents=True, exist_ok=True)
    for stem in present:
        (loras / f'{stem}.safetensors').write_bytes(b'x')
    from app import config as cfg
    cfg.save_config({'comfyui': {'base_dir': str(base)}})
    return base


def test_missing_row_is_dropped_and_the_rest_still_chains(app, tmp_path):
    from app.services import krea_edit_helper as keh
    _comfy_with_loras(tmp_path, present=('bypass', 'detail'))
    rows = keh._existing_generation_lora_rows([
        {'file': 'krea/bypass.safetensors', 'strength': 13.0},
        {'file': 'krea/gone.safetensors', 'strength': 1.0},
        {'file': 'krea/detail.safetensors', 'strength': 0.6},
    ])
    assert [r['file'] for r in rows] == ['krea/bypass.safetensors',
                                        'krea/detail.safetensors']


def test_zero_strength_row_is_dropped(app, tmp_path):
    from app.services import krea_edit_helper as keh
    _comfy_with_loras(tmp_path, present=('bypass',))
    assert keh._existing_generation_lora_rows(
        [{'file': 'krea/bypass.safetensors', 'strength': 0}]) == []


def test_rows_are_clamped_and_capped_at_enqueue_too(app, tmp_path):
    """The clamp is not only in the sanitizer: a caller could hand rows straight
    in, and 999 must never reach ComfyUI."""
    from app.services import krea_edit_helper as keh
    _comfy_with_loras(tmp_path, present=('bypass',))
    rows = keh._existing_generation_lora_rows(
        [{'file': 'krea/bypass.safetensors', 'strength': 999}] * 20)
    assert len(rows) == keh.MAX_GENERATION_LORAS
    assert all(r['strength'] == keh.LORA_STRENGTH_MAX for r in rows)


# --- Service and route -------------------------------------------------------

def test_fanout_applies_the_preset_to_every_variation(app, tmp_path, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.services import krea_edit_helper as keh
    seen = []
    monkeypatch.setattr(keh, 'preflight', lambda: None)
    monkeypatch.setattr(keh, 'enqueue_krea_edit',
                        lambda **kw: (seen.append(kw.get('generation_loras')), 'job')[1])
    _set_presets(PRESETS)
    with app.app_context():
        ds = _krea_dataset(app, svc)
        svc.generate_variations_krea(
            'local', ds.id,
            [{'label': 'a', 'prompt': 'p', 'framing': 'face'},
             {'label': 'b', 'prompt': 'q', 'framing': 'bust'}],
            1, generation_lora_preset='Bypass')
    assert len(seen) == 2
    assert all([r['file'] for r in rows] ==
               ['krea/krea2filterbypass3.safetensors', 'krea/detail_slider.safetensors']
               for rows in seen)


def test_fanout_without_a_preset_sends_no_rows(app, tmp_path, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.services import krea_edit_helper as keh
    seen = []
    monkeypatch.setattr(keh, 'preflight', lambda: None)
    monkeypatch.setattr(keh, 'enqueue_krea_edit',
                        lambda **kw: (seen.append(kw.get('generation_loras')), 'job')[1])
    _set_presets(PRESETS)
    with app.app_context():
        ds = _krea_dataset(app, svc)
        svc.generate_variations_krea('local', ds.id,
                                    [{'label': 'a', 'prompt': 'p', 'framing': 'face'}], 1)
    assert seen == [[]]


def test_generate_route_passes_the_krea_preset_name(app, client, tmp_path, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.services import krea_edit_helper as keh
    seen = {}
    # The route preflights Krea itself (before dispatch, so a mixed run never bills
    # an API batch only to discover Krea can't render its share) — bypass it here,
    # the point of this test is the request -> service argument wiring.
    monkeypatch.setattr(keh, 'preflight', lambda: None)
    monkeypatch.setattr(svc, 'generate_variations_krea',
                        lambda *a, **kw: (seen.update(kw), [1])[1])
    _set_presets(PRESETS)
    with app.app_context():
        ds_id = _krea_dataset(app, svc).id
    r = client.post(f'/api/dataset/{ds_id}/generate', json={
        'engine_batches': [{'generator': 'krea',
                            'variations': [{'label': 'a', 'prompt': 'p'}]}],
        'multiplier': 1,
        'krea_generation_lora_preset': 'Bypass',
        # The Klein key must NOT be read by the Krea branch — one run can carry both.
        'generation_lora_preset': 'Klein only',
    })
    assert r.status_code == 200, r.get_json()
    assert seen['generation_lora_preset'] == 'Bypass'
