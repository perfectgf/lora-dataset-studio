"""The Krea conditioning rebalance and the Krea2T enhancer are retired.

What is pinned, and why none of it is visible from a diff:

1. **The shipped Krea graphs name core ComfyUI classes only.** Node 30 lived in
   both templates, ON at x4, and "off" was the same node at identity — so a bare
   ComfyUI got a 409 "install ComfyUI-Conditioning-Rebalance" at launch with the
   toggle OFF. The template is loaded from disk here, not stubbed, because a stub
   would freeze the shape the file had the day this was written.
2. **The wire is deaf to the old fields, not allergic to them.** A browser tab
   open from before the change keeps sending `rebalance`, `rebalance_strength`,
   `enhancer`, `enhancer_strength`; the run must go ahead as if they were not
   there — an old tab breaking every launch would be the expensive kind of wrong.
3. **The builder has no such knob any more.** Not "ignores it": a kwarg nobody
   reads is how a feature comes back by accident. TypeError is the contract.
4. **The columns stay.** `krea_rebalance` and `enhancer_strength` are nullable
   and no longer written; a cell rendered with either keeps saying so in
   "Made with".
5. **The panel has no toggle**, read as text (node --test cannot mount JSX).

Why retired rather than kept opt-in: measured at a fixed seed on the
maintainer's own lane, the rebalance at x4 did not refine skin texture, it
re-decided the picture (52/255 mean pixel difference, 94% of pixels moved); the
enhancer had zero recorded use. The alias/pack-hint tables keep their entries
for the class, because the resolver is generic and a user template or a Canvas
plugin graph may still carry it — those are tested elsewhere.
"""
import json

import pytest

from app import config as cfg


def _graph(name):
    with open(cfg.BACKEND_DIR / 'workflows' / name, encoding='utf-8') as fh:
        return json.load(fh)


def _classes(wf):
    return [n.get('class_type') for n in wf.values() if isinstance(n, dict)]


# --- 1. The templates are bare ------------------------------------------------

@pytest.mark.parametrize('name', ['krea2_turbo.json', 'krea2_turbo_img2img.json'])
def test_the_shipped_templates_carry_no_rebalance_node(name):
    wf = _graph(name)
    assert '30' not in wf
    assert 'ConditioningKrea2Rebalance' not in _classes(wf)
    # The positive prompt feeds the sampler directly — nothing sits in between.
    assert wf['26']['inputs']['positive'] == ['23', 0]


def test_the_templates_name_only_core_classes():
    """The whole point: a fresh ComfyUI runs this without installing anything.
    Pinned as a list so a future template edit that reaches for a pack has to
    say so here, on purpose."""
    core = {'UNETLoader', 'CLIPLoader', 'VAELoader', 'CLIPTextEncode',
            'EmptySD3LatentImage', 'KSampler', 'VAEDecode', 'SaveImage',
            'LoadImage', 'VAEEncode'}
    for name in ('krea2_turbo.json', 'krea2_turbo_img2img.json'):
        assert set(_classes(_graph(name))) <= core, name


def _krea_builder(monkeypatch):
    from app.services import lora_test_studio as lts
    monkeypatch.setattr(lts, 'get_krea_loras', lambda: [{'filename': 'krea/t.safetensors'}])
    monkeypatch.setattr(lts, 'get_krea_models', lambda: [])
    monkeypatch.setattr(lts, 'krea_default_base', lambda: None)
    return lts


def test_the_built_cell_names_no_custom_class(app, monkeypatch):
    """Asserted on the graph the preflight scans and ComfyUI receives — this is
    the 409-on-a-fresh-install fix, at the seam where it matters."""
    lts = _krea_builder(monkeypatch)
    with app.app_context():
        wf = lts._build_cell_workflow('local', 'krea/t.safetensors', 1.0, 'p', 1, None,
                                      set(), train_type='krea')
    assert 'ConditioningKrea2Rebalance' not in _classes(wf)
    assert 'ComfyUI-Krea2T-Enhancer' not in _classes(wf)


# --- 2 + 3. The wire and the builder -----------------------------------------

def test_an_old_payload_still_sending_the_retired_fields_is_ignored_not_refused():
    from app.services.lora_test_studio import StudioGenSettings, _sanitize_gen_knobs
    s = StudioGenSettings.from_payload({'rebalance': True, 'rebalance_strength': 4.0,
                                        'enhancer': True, 'enhancer_strength': 1.5,
                                        'sampler': 'euler'})
    for gone in ('rebalance', 'rebalance_strength', 'enhancer', 'enhancer_strength'):
        assert not hasattr(s, gone), gone
    assert s.sampler == 'euler'
    knobs = _sanitize_gen_knobs('krea')
    assert 'enhancer_strength' not in knobs and 'rebalance' not in knobs


def test_the_builder_refuses_the_retired_kwargs(app, monkeypatch):
    """A kwarg nobody reads is how a feature comes back by accident."""
    lts = _krea_builder(monkeypatch)
    with app.app_context():
        for gone in ({'rebalance': 4.0}, {'enhancer_strength': 1.0}):
            with pytest.raises(TypeError):
                lts._build_cell_workflow('local', 'krea/t.safetensors', 1.0, 'p', 1, None,
                                         set(), train_type='krea', **gone)


def test_the_injections_no_longer_exist():
    import app.utils.comfyui as comfyui
    for gone in ('inject_krea_rebalance', 'inject_krea2t_enhancer',
                 'KREA_REBALANCE_CLASS', 'KREA2T_ENHANCER_CLASS'):
        assert not hasattr(comfyui, gone), gone


# --- 4. The columns stay ------------------------------------------------------

def test_the_legacy_columns_stay_nullable_and_unwritten(app, monkeypatch):
    """A cell created today writes NULL to both; a cell from before keeps its value."""
    from app.models import LoraTestImage
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Krea', 'kt')
        ck = 'krea\\lora_kt_000001000.safetensors'
        monkeypatch.setattr(lts, 'list_test_checkpoints', lambda _ds, _f=None: [{'filename': ck}])
        monkeypatch.setattr(lts, 'get_krea_models', lambda: [])
        monkeypatch.setattr(lts, 'permanent_lora_candidates', lambda _f: [])
        monkeypatch.setattr(lts, 'gpu_busy_reason', lambda: None)
        monkeypatch.setattr(lts, '_active_run_count', lambda *a: 0)
        monkeypatch.setattr(lts, '_preflight_checkpoint_arch', lambda *a, **k: None)
        monkeypatch.setattr(lts, '_preflight_run', lambda *a, **k: None)
        monkeypatch.setattr(lts, '_target_node_classes', lambda: None)
        monkeypatch.setattr(lts, '_build_cell_workflow', lambda *a, **k: {'1': {}})
        monkeypatch.setattr(lts, '_enqueue_cell', lambda *a, job_id=None, **k: job_id)

        out = lts.create_run(LOCAL_USER, ds.id, [ck], [1.0],
                             lts.StudioGenSettings(prompt='p', count=1), family='krea')
        rows = LoraTestImage.query.filter_by(run_id=out['run_id']).all()
        assert rows and all(r.krea_rebalance is None and r.enhancer_strength is None
                            for r in rows)
        # The columns still exist for what history recorded.
        assert hasattr(LoraTestImage, 'krea_rebalance')
        assert hasattr(LoraTestImage, 'enhancer_strength')


# --- 5. The panel -------------------------------------------------------------

def test_the_panel_has_neither_toggle():
    src = (cfg.BACKEND_DIR.parent / 'frontend' / 'src' / 'components' / 'dataset'
           / 'studio' / 'StudioGenerationSettings.jsx').read_text(encoding='utf-8')
    low = src.lower()
    assert 'rebalance' not in low
    assert 'krea2t' not in low and 'enhancer' not in low
