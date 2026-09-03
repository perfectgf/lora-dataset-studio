"""🧬 Blend from the ◉ LoRA Canvas — every ticked checkpoint in ONE generation.

The board used to run one pass per pick. Blend is the Test Studio's Combine mode
driven from the board: the picks are loaded together, each at its own weight, and
every dataset's trigger word is injected into the prompt.

What is pinned here is the CHAIN, end to end, and not just the flag:

  · the route forwards `combine` and the per-selection `weight` untouched — a
    dropped flag degrades a blend into a comparison grid in silence, which looks
    like "it worked" until you count the images;
  · the workflow ACTUALLY SUBMITTED to ComfyUI carries N chained LoRA loaders at
    their own strengths and both triggers in the text encoder. That assertion is
    made on the real `_build_cell_workflow` output captured at the queue's door,
    not on the arguments handed to a mock;
  · the canvas index carries each dataset's trigger word, which is what lets the
    panel NAME what it is about to inject instead of doing it silently.
"""
import pytest

_ST = (b'\x08\x00\x00\x00\x00\x00\x00\x00{"__metadata__":{}}'
       .ljust(32, b'\x00'))


def _comfy(monkeypatch, reachable=True):
    monkeypatch.setattr('app.capabilities.probe',
                        lambda *a, **k: {'comfyui': {'reachable': reachable}})
    # The launch gate asks the ComfyUI probe alone — the runner has no ComfyUI.
    monkeypatch.setattr('app.capabilities.probe_comfyui',
                        lambda: {'ok': reachable, 'status': 'ok' if reachable else 'unreachable',
                                 'detail': '', 'hint': ''})


def _zimage_tree(tmp_path, monkeypatch, loras):
    """A configured ComfyUI tree holding `loras` (bare file names) + one UNET."""
    from app import config
    base = tmp_path / 'Comfy'
    lora_dir = base / 'models' / 'loras' / 'z image'
    lora_dir.mkdir(parents=True, exist_ok=True)
    for name in loras:
        (lora_dir / name).write_bytes(_ST)
    unet_dir = base / 'models' / 'unet' / 'z image'
    unet_dir.mkdir(parents=True, exist_ok=True)
    (unet_dir / 'zmodel.safetensors').write_bytes(_ST)
    config.save_config({'comfyui': {'base_dir': str(base)}})
    import app.utils.comfyui as comfyui_utils
    monkeypatch.setattr(comfyui_utils, '_zimage_models_cache',
                        {'data': None, 'timestamp': 0})
    return base


# --- the route is a pass-through ---------------------------------------------

def test_canvas_generate_forwards_combine_and_every_weight(client, monkeypatch):
    """`combine` and each selection's `weight` must reach the engine untouched."""
    _comfy(monkeypatch)
    seen = {}

    def fake(user_id, selections, strengths, settings=None, **kwargs):
        seen['selections'] = selections
        seen['combine'] = kwargs.get('combine')
        seen['strengths'] = strengths
        return {'created': 1, 'seed': 7, 'count': 1, 'run_id': 'r1', 'ids': []}

    monkeypatch.setattr('app.services.cloud_training.canvas_generate', fake)
    resp = client.post('/api/train/canvas/generate', json={
        'selections': [
            {'dataset_id': 1, 'checkpoint': 'a.safetensors',
             'record_id': 10, 'step': 1000, 'weight': 0.9},
            {'dataset_id': 2, 'checkpoint': 'b.safetensors',
             'record_id': 20, 'step': 2000, 'weight': 0.55},
        ],
        'combine': True, 'count': 1})
    assert resp.status_code == 200
    assert seen['combine'] is True
    assert [s.get('weight') for s in seen['selections']] == [0.9, 0.55]
    # The board sends no strength axis in a blend; the route must not invent one
    # other than the harmless default the engine replaces anyway.
    assert seen['strengths'] == [1.0]


def test_canvas_generate_without_combine_is_unchanged(client, monkeypatch):
    """A Compare launch keeps its historical body: no `combine`, sweep intact."""
    _comfy(monkeypatch)
    seen = {}

    def fake(user_id, selections, strengths, settings=None, **kwargs):
        seen.update(kwargs)
        seen['strengths'] = strengths
        return {'created': 2, 'seed': 7, 'count': 1, 'run_id': 'r1', 'ids': []}

    monkeypatch.setattr('app.services.cloud_training.canvas_generate', fake)
    resp = client.post('/api/train/canvas/generate', json={
        'selections': [{'dataset_id': 1, 'checkpoint': 'a.safetensors'}],
        'strengths': [0.8, 1.0]})
    assert resp.status_code == 200
    assert seen['combine'] is None
    assert seen['strengths'] == [0.8, 1.0]


# --- THE proof: what really goes to ComfyUI ----------------------------------

def test_canvas_blend_submits_one_workflow_chaining_every_lora_at_its_weight(
        app, monkeypatch, tmp_path):
    """End of the chain, on the workflow the queue actually receives.

    Two checkpoints ticked on the board, blended at 0.9 and 0.55: ONE cell, whose
    submitted graph holds two CHAINED LoraLoaderModelOnly nodes at exactly those
    strengths, with both datasets' triggers prefixed to the encoded prompt. Only
    `_enqueue_cell` is replaced — the workflow itself is built for real."""
    from app.services import cloud_training as ct, lora_test_studio as lts
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app.models import LoraTestImage
    with app.app_context():
        name_a, name_b = 'lora_aaa_000002000.safetensors', 'lora_bbb_000001000.safetensors'
        _zimage_tree(tmp_path, monkeypatch, [name_a, name_b])
        cp_a = 'z image' + chr(92) + name_a
        cp_b = 'z image' + chr(92) + name_b
        ds_a = svc.create_dataset(LOCAL_USER, 'Alpha', 'aaa')
        ds_b = svc.create_dataset(LOCAL_USER, 'Beta', 'bbb')
        by_ds = {ds_a.id: [{'filename': cp_a}], ds_b.id: [{'filename': cp_b}]}
        monkeypatch.setattr(lts, 'list_test_checkpoints',
                            lambda ds, _family=None: by_ds[ds.id])
        monkeypatch.setattr(lts, 'gpu_busy_reason', lambda: None)
        monkeypatch.setattr(lts, '_active_run_count', lambda *a: 0)
        monkeypatch.setattr(lts, '_preflight_checkpoint_arch', lambda *a, **k: None)
        monkeypatch.setattr(lts, '_preflight_run', lambda *a, **k: None)
        monkeypatch.setattr(lts, '_target_node_classes', lambda: None)
        monkeypatch.setattr(lts, 'permanent_lora_candidates', lambda _f: [])
        submitted = []

        def capture(user_id, dataset_id, workflow, prompt, job_id=None, **_kw):
            submitted.append(workflow)
            return job_id
        monkeypatch.setattr(lts, '_enqueue_cell', capture)

        out = ct.canvas_generate(
            LOCAL_USER,
            [{'dataset_id': ds_a.id, 'checkpoint': cp_a,
              'record_id': 11, 'step': 2000, 'weight': 0.9},
             {'dataset_id': ds_b.id, 'checkpoint': cp_b,
              'record_id': 22, 'step': 1000, 'weight': 0.55}],
            [0.6, 0.8, 1.0],               # the sweep a blend has no use for
            lts.StudioGenSettings(prompt='on a rooftop', count=1), combine=True)

        # ONE generation, not one per pick and per strength.
        assert out['created'] == 1
        assert len(submitted) == 1
        wf = submitted[0]

        loaders = {nid: n for nid, n in wf.items()
                   if n.get('class_type') == 'LoraLoaderModelOnly'}
        assert {(n['inputs']['lora_name'], n['inputs']['strength_model'])
                for n in loaders.values()} == {(cp_a, 0.9), (cp_b, 0.55)}
        # Chained, not parallel: exactly one loader is fed by another loader, so
        # both LoRAs really stack on the same model path.
        fed_by_loader = [n for n in loaders.values()
                         if n['inputs']['model'][0] in loaders]
        assert len(fed_by_loader) == 1
        # Both triggers reach the text encoder, head LoRA's first.
        texts = [n['inputs'].get('text') for n in wf.values()
                 if isinstance(n.get('inputs', {}).get('text'), str)]
        assert any(t.startswith('aaa, bbb, on a rooftop') for t in texts)

        # …and the persisted cell remembers the stack, so the board can say what
        # the picture was made of instead of showing an anonymous image.
        row = LoraTestImage.query.filter_by(run_id=out['run_id']).one()
        assert row.checkpoint == cp_a and row.strength == 0.9
        assert row.record_id == 11 and row.step == 2000
        members = lts.stack_of_row(row)
        assert [(m['filename'], m['weight'], m['trigger']) for m in members] == [
            (cp_a, 0.9, 'aaa'), (cp_b, 0.55, 'bbb')]


def test_canvas_blend_refuses_to_mix_families_and_names_them(app, monkeypatch):
    """Belt and braces with the greyed-out toggle: the engine refuses too, and
    the reason travels back to the button rather than a bare 400."""
    from app.services import cloud_training as ct, lora_test_studio as lts
    from app.config import LOCAL_USER
    with app.app_context():
        monkeypatch.setattr(lts, 'gpu_busy_reason', lambda: None)
        monkeypatch.setattr(lts, '_active_run_count', lambda *a: 0)
        with pytest.raises(ValueError) as excinfo:
            ct.canvas_generate(
                LOCAL_USER,
                [{'dataset_id': 1, 'checkpoint': 'krea' + chr(92) + 'a.safetensors'},
                 {'dataset_id': 2, 'checkpoint': 'sdxl' + chr(92) + 'b.safetensors'}],
                [1.0], combine=True)
        assert 'one family per run' in str(excinfo.value)


# --- what lets the panel NAME the triggers before injecting them -------------

def test_canvas_index_carries_each_datasets_trigger_word(app):
    """The board's blend panel lists the words it is about to prepend. It can
    only do that if the cheap index it already fetches carries them."""
    from app.extensions import db
    from app.models import FaceDataset, TrainingRunRecord
    from app.services import cloud_training as ct
    with app.app_context():
        with_trigger = FaceDataset(user_id='local', name='Alpha', trigger_word='aaa')
        without = FaceDataset(user_id='local', name='Beta', trigger_word='')
        db.session.add_all([with_trigger, without])
        db.session.commit()
        for ds in (with_trigger, without):
            db.session.add(TrainingRunRecord(
                dataset_id=ds.id, family='zimage', source='local', base_model='',
                variant='turbo', steps=1000, version=1, fingerprint='fp', manifest='[]'))
        db.session.commit()
        rows = {d['name']: d for d in ct.canvas_dataset_index('local')['datasets']}
        assert rows['Alpha']['trigger_word'] == 'aaa'
        # An empty trigger travels as None, never '' — the panel branches on it.
        assert rows['Beta']['trigger_word'] is None
