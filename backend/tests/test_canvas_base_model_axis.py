"""◉ Canvas — the BASE MODEL is an axis, like it already is in the Test Studio.

The board's panel offers « BASE MODEL (MULTI) » and lets several bases be
ticked, but the launch only ever ran ONE of them: the canvas sent
`zModels[0]` and `create_comparison_run` pinned a single `z_model` before it
built its matrix. Three bases ticked therefore produced one generation, and
nothing said the other two had been dropped — the worst shape a limit can take,
because the run LOOKS like the one that was asked for.

`create_run` (one dataset, the Test Studio) has always swept a `z_models` list.
This pins the same contract on the comparison engine the canvas drives, plus
the two properties a sweep gets wrong in silence:

  · ADDITIVITY — a caller that only knows `z_model` produces exactly the run it
    always produced. Every existing surface and every older frontend keeps
    working, which is what makes this safe to land in a shared engine;
  · the PRODUCT — N bases × the cells = N × the cells, in ONE run, each row
    carrying the base it was actually generated with. A sweep that mislabels
    its rows is worse than no sweep: the images all arrive, all look
    plausible, and the comparison they were fired for is a lie.
"""

_ST = (b'\x08\x00\x00\x00\x00\x00\x00\x00{"__metadata__":{}}'.ljust(32, b'\x00'))


def _tree_with_two_bases(tmp_path, monkeypatch, loras):
    """A ComfyUI tree carrying TWO Z-Image bases, so the axis has something to
    sweep. Everything else matches the other studio fixtures."""
    from app import config
    base = tmp_path / 'Comfy'
    lora_dir = base / 'models' / 'loras' / 'z image'
    lora_dir.mkdir(parents=True, exist_ok=True)
    for name in loras:
        (lora_dir / name).write_bytes(_ST)
    unet = base / 'models' / 'unet' / 'z image'
    unet.mkdir(parents=True, exist_ok=True)
    (unet / 'base_one.safetensors').write_bytes(_ST)
    (unet / 'base_two.safetensors').write_bytes(_ST)
    config.save_config({'comfyui': {'base_dir': str(base)}})
    import app.utils.comfyui as comfyui_utils
    monkeypatch.setattr(comfyui_utils, '_zimage_models_cache',
                        {'data': None, 'timestamp': 0})


def _canvas_launch(app_ctx_tmp, monkeypatch):
    """(launch, models) — two datasets ticked on the board, ready to fire.

    Two datasets because that IS the canvas case: the board's whole point is
    comparing checkpoints that do not share a dataset, and that is the path
    which routes through `create_comparison_run`.
    """
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.config import LOCAL_USER
    tmp_path = app_ctx_tmp
    name_a, name_b = 'lora_aaa_000002000.safetensors', 'lora_bbb_000001000.safetensors'
    _tree_with_two_bases(tmp_path, monkeypatch, [name_a, name_b])
    cp_a = 'z image' + chr(92) + name_a
    cp_b = 'z image' + chr(92) + name_b
    ds_a = svc.create_dataset(LOCAL_USER, 'Alpha', 'aaa')
    ds_b = svc.create_dataset(LOCAL_USER, 'Beta', 'bbb')
    by_ds = {ds_a.id: [{'filename': cp_a}], ds_b.id: [{'filename': cp_b}]}
    monkeypatch.setattr(lts, 'list_test_checkpoints', lambda ds, _f=None: by_ds[ds.id])
    monkeypatch.setattr(lts, 'gpu_busy_reason', lambda: None)
    monkeypatch.setattr(lts, '_active_run_count', lambda *a: 0)
    monkeypatch.setattr(lts, '_preflight_checkpoint_arch', lambda *a, **k: None)
    monkeypatch.setattr(lts, '_preflight_run', lambda *a, **k: None)
    monkeypatch.setattr(lts, '_target_node_classes', lambda: None)
    monkeypatch.setattr(lts, 'permanent_lora_candidates', lambda _f: [])
    monkeypatch.setattr(lts, '_build_cell_workflow', lambda *a, **k: {'1': {}})
    monkeypatch.setattr(lts, '_enqueue_cell', lambda *a, job_id=None, **k: job_id)

    from app.utils.comfyui import get_zimage_models
    models = get_zimage_models()

    def launch(**kw):
        return lts.create_comparison_run(
            LOCAL_USER,
            [{'dataset_id': ds_a.id, 'checkpoint': cp_a},
             {'dataset_id': ds_b.id, 'checkpoint': cp_b}],
            kw.pop('strengths', [1.0]),
            lts.StudioGenSettings(prompt='on a rooftop', **kw))

    return launch, models


def test_two_ticked_bases_generate_on_both(app, monkeypatch, tmp_path):
    """The product: 2 bases × 2 checkpoints = 4 cells, one run, each row
    carrying the base it was really made with."""
    from app.models import LoraTestImage
    with app.app_context():
        launch, models = _canvas_launch(tmp_path, monkeypatch)
        assert len(models) == 2, f'fixture should offer two bases, got {models}'

        out = launch(z_models=list(models))

        assert out['created'] == 4
        rows = LoraTestImage.query.filter_by(run_id=out['run_id']).all()
        assert sorted({r.z_model for r in rows}) == sorted(models)
        # Every base ran on every ticked checkpoint — a sweep that drops a
        # pairing still "works", it just answers a narrower question.
        assert len({(r.z_model, r.checkpoint) for r in rows}) == 4


def test_a_single_base_model_is_the_run_it_always_was(app, monkeypatch, tmp_path):
    """Additivity: a caller that only knows `z_model` is untouched."""
    from app.models import LoraTestImage
    with app.app_context():
        launch, models = _canvas_launch(tmp_path, monkeypatch)

        out = launch(z_model=models[1])

        assert out['created'] == 2
        rows = LoraTestImage.query.filter_by(run_id=out['run_id']).all()
        assert {r.z_model for r in rows} == {models[1]}
