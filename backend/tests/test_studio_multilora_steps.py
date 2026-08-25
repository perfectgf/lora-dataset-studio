"""🎛 Steps (and CFG) are settable with SEVERAL LoRAs selected — compare AND blend.

User report: "I cannot set the number of steps when I have two LoRAs selected, in
blend or in comparison mode."

The field was not disabled and it was not overwritten: it was ABSENT. The
multi-LoRA branch of the Test Studio (`ComparisonStudio` → `StudioRunSetup`) had
no CFG/steps picker at all, because those ladders reach the single-LoRA studio and
the canvas panel through a per-DATASET payload the comparison has no dataset for.
Its launch body therefore never carried `steps`, and every cell fell back to the
family default.

The route and the engine were already able to do it — which is what makes the fix
a front-end one plus one additive field on `/api/studio/base-models`. What is
pinned here is the half that can regress silently:

  · the ladders are published without a dataset, and are the SAME constants the
    dataset payload sends (two ladders would be two answers to one question);
  · non-default steps really reach the WORKFLOW SUBMITTED to ComfyUI, in
    comparison AND in blend — asserted on the graph captured at the queue's door,
    not on the arguments handed to a mock;
  · the value is persisted on the cell, so a resume replays the run that was
    launched rather than the default one.
"""
from test_studio_prompt_batch import _comfy, _neutralise_preflights, _zimage_tree

NON_DEFAULT_STEPS = 24          # not DEFAULT_STEPS (8)
NON_DEFAULT_CFG = 3.5           # not DEFAULT_CFG (1.0)


def _sampler_steps(workflow):
    """Every `steps` input of the submitted graph, whatever node carries it."""
    return [n['inputs']['steps'] for n in workflow.values()
            if isinstance(n, dict) and isinstance(n.get('inputs'), dict)
            and 'steps' in n['inputs']]


# --- the ladders exist without a dataset -------------------------------------

def test_base_models_publishes_the_axes_ladders(client, monkeypatch):
    """The comparison has no dataset; the ladders must come with the bases."""
    from app.services import lora_test_studio as lts
    _comfy(monkeypatch)
    resp = client.get('/api/studio/base-models?type=zimage')
    assert resp.status_code == 200
    axes = resp.get_json()['axes']
    # The SAME constants the per-dataset payload sends — not a second ladder.
    assert axes['steps_choices'] == lts.STEPS_CHOICES
    assert axes['default_steps'] == lts.DEFAULT_STEPS
    assert axes['cfg_choices'] == lts.CFG_CHOICES
    assert axes['default_cfg'] == lts.DEFAULT_CFG
    # The second pass belongs to the SDXL workflow only.
    assert axes['steps2_choices'] is None
    sdxl = client.get('/api/studio/base-models?type=sdxl').get_json()['axes']
    assert sdxl['steps2_choices'] == lts.STEPS_CHOICES
    # `models` keeps its exact shape: an older frontend is unaffected.
    assert isinstance(resp.get_json()['models'], list)


def test_the_run_route_forwards_steps_and_cfg(client, monkeypatch):
    _comfy(monkeypatch)
    seen = {}

    def fake(user_id, selections, strengths, settings=None, **kwargs):
        seen.update(kwargs)
        seen['steps_list'] = settings.steps_list if settings else None
        seen['cfgs'] = settings.cfgs if settings else None
        return {'created': 1, 'seed': 7, 'count': 1, 'run_id': 'r1', 'ids': []}

    monkeypatch.setattr('app.services.lora_test_studio.create_comparison_run', fake)
    resp = client.post('/api/studio/run', json={
        'selections': [{'dataset_id': 1, 'checkpoint': 'a.safetensors'}],
        'strengths': [1.0], 'steps': [NON_DEFAULT_STEPS], 'cfgs': [NON_DEFAULT_CFG]})
    assert resp.status_code == 200
    assert seen['steps_list'] == [NON_DEFAULT_STEPS]
    assert seen['cfgs'] == [NON_DEFAULT_CFG]


# --- THE proof: the graph ComfyUI receives -----------------------------------

def _two_lora_run(app, monkeypatch, tmp_path, suffix, **knobs):
    """Launch a two-LoRA comparison/blend for real and return (out, workflows)."""
    from app.services import lora_test_studio as lts
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    name_a = f'lora_{suffix}a_000002000.safetensors'
    name_b = f'lora_{suffix}b_000001000.safetensors'
    _zimage_tree(tmp_path, monkeypatch, [name_a, name_b])
    cp_a = 'z image' + chr(92) + name_a
    cp_b = 'z image' + chr(92) + name_b
    ds_a = svc.create_dataset(LOCAL_USER, f'A{suffix}', f'{suffix}a')
    ds_b = svc.create_dataset(LOCAL_USER, f'B{suffix}', f'{suffix}b')
    by_ds = {ds_a.id: [{'filename': cp_a}], ds_b.id: [{'filename': cp_b}]}
    _neutralise_preflights(monkeypatch, lts)
    monkeypatch.setattr(lts, 'list_test_checkpoints',
                        lambda ds, _family=None: by_ds[ds.id])
    submitted = []

    def capture(user_id, dataset_id, workflow, prompt, job_id=None, **_kw):
        submitted.append(workflow)
        return job_id
    monkeypatch.setattr(lts, '_enqueue_cell', capture)
    out = lts.create_comparison_run(
        LOCAL_USER,
        [{'dataset_id': ds_a.id, 'checkpoint': cp_a, 'weight': 0.9},
         {'dataset_id': ds_b.id, 'checkpoint': cp_b, 'weight': 0.6}],
        [1.0],
        lts.StudioGenSettings(seed=11, prompt='on a rooftop', count=1,
                              **{k: v for k, v in knobs.items() if k != 'combine'}),
        combine=knobs.get('combine'))
    return out, submitted


def test_comparison_run_submits_the_steps_that_were_set(app, monkeypatch, tmp_path):
    """Two LoRAs, ⚖ Compare, steps 24: both graphs sample 24 times, not 8."""
    from app.models import LoraTestImage
    with app.app_context():
        out, submitted = _two_lora_run(
            app, monkeypatch, tmp_path, 'cmp',
            steps_list=[NON_DEFAULT_STEPS], cfgs=[NON_DEFAULT_CFG])
        assert out['created'] == 2 and len(submitted) == 2
        for wf in submitted:
            steps = _sampler_steps(wf)
            assert steps, 'the submitted graph carries no steps input at all'
            assert set(steps) == {NON_DEFAULT_STEPS}
        rows = LoraTestImage.query.filter_by(run_id=out['run_id']).all()
        # Persisted, so a resume replays THIS run and not the default one.
        assert {r.steps for r in rows} == {NON_DEFAULT_STEPS}
        assert {r.cfg for r in rows} == {NON_DEFAULT_CFG}


def test_blend_run_submits_the_steps_that_were_set(app, monkeypatch, tmp_path):
    """Same two LoRAs, 🧬 Blend (one image, both loaded): the strength axis is
    gone but the steps axis is not — steps are a RENDER setting, not a LoRA one."""
    from app.models import LoraTestImage
    with app.app_context():
        out, submitted = _two_lora_run(
            app, monkeypatch, tmp_path, 'bld',
            combine=True, steps_list=[NON_DEFAULT_STEPS], cfgs=[NON_DEFAULT_CFG])
        assert out['created'] == 1 and len(submitted) == 1
        steps = _sampler_steps(submitted[0])
        assert steps and set(steps) == {NON_DEFAULT_STEPS}
        row = LoraTestImage.query.filter_by(run_id=out['run_id']).one()
        assert row.steps == NON_DEFAULT_STEPS and row.cfg == NON_DEFAULT_CFG


def test_a_steps_sweep_is_still_a_sweep_in_blend(app, monkeypatch, tmp_path):
    """Several steps values = several cells, in blend too — the axis multiplies
    the run exactly like everywhere else, which is what the cost counter says."""
    from app.models import LoraTestImage
    with app.app_context():
        out, submitted = _two_lora_run(
            app, monkeypatch, tmp_path, 'swp',
            combine=True, steps_list=[8, NON_DEFAULT_STEPS])
        assert out['created'] == 2 and len(submitted) == 2
        rows = LoraTestImage.query.filter_by(run_id=out['run_id']).all()
        assert sorted(r.steps for r in rows) == [8, NON_DEFAULT_STEPS]


def test_no_steps_sent_keeps_the_familys_default(app, monkeypatch, tmp_path):
    """Zero regression for a client that sends nothing: the historical default."""
    from app.services import lora_test_studio as lts
    from app.models import LoraTestImage
    with app.app_context():
        out, _ = _two_lora_run(app, monkeypatch, tmp_path, 'def')
        rows = LoraTestImage.query.filter_by(run_id=out['run_id']).all()
        assert {r.steps for r in rows} == {lts.DEFAULT_STEPS}
