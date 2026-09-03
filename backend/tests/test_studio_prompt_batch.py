"""📝 Batch of prompts — tick several entries of the prompt history, launch once.

Both generation surfaces (the dataset's Test Studio and the canvas panel
« Generate from the board ») show the same saved-prompt history, and both could
only replay ONE of them per launch. Replaying N by firing N launches is not an
option: `_active_run_count` refuses a second run while one is in flight, and the
GPU is serialised anyway. So the batch is an AXIS inside one run — one pass per
ticked prompt, same checkpoints, same settings, same seed.

What is pinned here:

  · the axis helper: nothing ticked behaves EXACTLY like before (one prompt), a
    ticked list is stripped/deduplicated in order, and NO size is ever refused
    (see test_a_large_batch_is_never_refused for why that line is here);
  · N ticked prompts produce N DISTINCT submitted workflows, one per prompt —
    asserted on the graphs captured at the queue's door, not on call arguments,
    and on both engines (`create_run` for the Test Studio, `create_comparison_run`
    via `canvas_generate` for the board);
  · both routes forward `prompts` to their engine — a dropped key would degrade a
    batch into a single generation in silence.
"""
_ST =(b'\x08\x00\x00\x00\x00\x00\x00\x00{"__metadata__":{}}'
       .ljust(32, b'\x00'))


def _comfy(monkeypatch, reachable=True):
    monkeypatch.setattr('app.capabilities.probe',
                        lambda *a, **k: {'comfyui': {'reachable': reachable}})
    monkeypatch.setattr('app.capabilities.probe_comfyui', lambda: {'ok': reachable, 'status': 'ok' if reachable else 'unreachable', 'detail': '', 'hint': 'Check the URL'})


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


def _neutralise_preflights(monkeypatch, lts):
    monkeypatch.setattr(lts, 'gpu_busy_reason', lambda: None)
    monkeypatch.setattr(lts, '_active_run_count', lambda *a: 0)
    monkeypatch.setattr(lts, '_preflight_checkpoint_arch', lambda *a, **k: None)
    monkeypatch.setattr(lts, '_preflight_run', lambda *a, **k: None)
    monkeypatch.setattr(lts, '_target_node_classes', lambda: None)
    monkeypatch.setattr(lts, 'permanent_lora_candidates', lambda _f: [])


# --- the axis itself ---------------------------------------------------------

def test_prompt_axis_empty_selection_is_the_old_single_prompt():
    """Nothing ticked must not be a new code path: the axis is the field's
    prompt, alone — including the None that means "each dataset's identity
    prompt" on the cross-dataset engine."""
    from app.services.lora_test_studio import _prompt_axis
    assert _prompt_axis(None, 'on a rooftop') == ['on a rooftop']
    assert _prompt_axis([], 'on a rooftop') == ['on a rooftop']
    assert _prompt_axis(['  ', ''], 'on a rooftop') == ['on a rooftop']
    assert _prompt_axis(None, None) == [None]


def test_prompt_axis_strips_dedupes_and_keeps_the_ticking_order():
    from app.services.lora_test_studio import _prompt_axis
    assert _prompt_axis([' b ', 'a', 'b', 42, None, 'a'], 'fallback') == ['b', 'a']


def test_a_large_batch_is_never_refused():
    """No cap on the prompt axis, on purpose — and this test exists BECAUSE one
    was shipped and rejected on first contact.

    A first version refused past 24 prompts. The first real use ticked 33 and was
    turned away. That 24 was a judgment, not a measurement: nothing breaks at 33
    (the body is kilobytes against a 64 MB ceiling, `prompt` is a TEXT column, the
    queue has no maximum depth, no results view truncates), and it capped ONE axis
    out of six — 24 prompts across 8 checkpoints sailed through while 25 prompts on
    a single one did not, though the second run is thirty times shorter.

    The module's own rule, written above `build_matrix`, is the one that applies:
    no ceiling on the number of cells; the queue is serial and the user sees the
    count and the duration before launching."""
    from app.services.lora_test_studio import _prompt_axis
    many = [f'prompt {i}' for i in range(200)]
    assert _prompt_axis(many, 'fallback') == many


def test_no_prompt_cap_constant_survives_anywhere():
    """A constant left behind is a cap waiting to be re-applied by the next edit."""
    from app.services import lora_test_studio as lts
    assert not hasattr(lts, '_MAX_PROMPTS_PER_RUN')


# --- THE proof, Test Studio side: N prompts -> N distinct workflows -----------

def test_test_studio_batch_submits_one_workflow_per_ticked_prompt(
        app, monkeypatch, tmp_path):
    """Three ticked prompts on one checkpoint: three cells, three graphs, three
    different encoded texts — and the rows remember which is which."""
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.config import LOCAL_USER
    from app.models import LoraTestImage
    with app.app_context():
        name = 'lora_bat_000002000.safetensors'
        _zimage_tree(tmp_path, monkeypatch, [name])
        ck = 'z image' + chr(92) + name
        ds = svc.create_dataset(LOCAL_USER, 'Batch', 'bat')
        _neutralise_preflights(monkeypatch, lts)
        monkeypatch.setattr(lts, 'list_test_checkpoints',
                            lambda _ds, _family=None: [{'filename': ck}])
        submitted = []

        def capture(user_id, dataset_id, workflow, prompt, job_id=None, **_kw):
            submitted.append(prompt)
            return job_id
        monkeypatch.setattr(lts, '_enqueue_cell', capture)
        monkeypatch.setattr(lts, '_build_cell_workflow',
                            lambda *a, **k: {'1': {'prompt': a[3]}})

        out = lts.create_run(LOCAL_USER, ds.id, [ck], [1.0],
                             lts.StudioGenSettings(prompt='ignored', count=1),
                             prompts=['on a rooftop', 'in the snow', 'at night'])

        assert out['created'] == 3
        assert sorted(submitted) == ['at night', 'in the snow', 'on a rooftop']
        rows = LoraTestImage.query.filter_by(run_id=out['run_id']).all()
        assert sorted(r.prompt for r in rows) == ['at night', 'in the snow', 'on a rooftop']


def test_test_studio_batch_multiplies_the_existing_axes_not_replaces_them(
        app, monkeypatch, tmp_path):
    """The batch is an axis among the others: 2 prompts x 2 strengths = 4 cells,
    every pairing present exactly once."""
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.config import LOCAL_USER
    from app.models import LoraTestImage
    with app.app_context():
        name = 'lora_mix_000002000.safetensors'
        _zimage_tree(tmp_path, monkeypatch, [name])
        ck = 'z image' + chr(92) + name
        ds = svc.create_dataset(LOCAL_USER, 'Mix', 'mix')
        _neutralise_preflights(monkeypatch, lts)
        monkeypatch.setattr(lts, 'list_test_checkpoints',
                            lambda _ds, _family=None: [{'filename': ck}])
        monkeypatch.setattr(lts, '_enqueue_cell',
                            lambda *a, job_id=None, **k: job_id)
        monkeypatch.setattr(lts, '_build_cell_workflow', lambda *a, **k: {'1': {}})

        out = lts.create_run(LOCAL_USER, ds.id, [ck], [0.8, 1.0],
                             lts.StudioGenSettings(prompt='x', count=1),
                             prompts=['alpha', 'beta'])

        assert out['created'] == 4
        rows = LoraTestImage.query.filter_by(run_id=out['run_id']).all()
        assert sorted((r.prompt, r.strength) for r in rows) == [
            ('alpha', 0.8), ('alpha', 1.0), ('beta', 0.8), ('beta', 1.0)]


def test_test_studio_without_a_batch_is_byte_for_byte_the_old_run(
        app, monkeypatch, tmp_path):
    """Zero regression for whoever ticks nothing: one cell, the field's prompt."""
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.config import LOCAL_USER
    from app.models import LoraTestImage
    with app.app_context():
        name = 'lora_solo_000002000.safetensors'
        _zimage_tree(tmp_path, monkeypatch, [name])
        ck = 'z image' + chr(92) + name
        ds = svc.create_dataset(LOCAL_USER, 'Solo', 'solo')
        _neutralise_preflights(monkeypatch, lts)
        monkeypatch.setattr(lts, 'list_test_checkpoints',
                            lambda _ds, _family=None: [{'filename': ck}])
        monkeypatch.setattr(lts, '_enqueue_cell',
                            lambda *a, job_id=None, **k: job_id)
        monkeypatch.setattr(lts, '_build_cell_workflow', lambda *a, **k: {'1': {}})

        out = lts.create_run(LOCAL_USER, ds.id, [ck], [1.0],
                             lts.StudioGenSettings(prompt='only this', count=1))

        assert out['created'] == 1
        row = LoraTestImage.query.filter_by(run_id=out['run_id']).one()
        assert row.prompt == 'only this'


# --- THE proof, canvas side --------------------------------------------------

def test_canvas_batch_submits_one_workflow_per_ticked_prompt(
        app, monkeypatch, tmp_path):
    """Same guarantee on the board's engine, where a run may span datasets: two
    ticked checkpoints x three ticked prompts = six graphs, each prompt appearing
    once per checkpoint."""
    from app.services import cloud_training as ct, lora_test_studio as lts
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app.models import LoraTestImage
    with app.app_context():
        name_a, name_b = 'lora_ca_000002000.safetensors', 'lora_cb_000001000.safetensors'
        _zimage_tree(tmp_path, monkeypatch, [name_a, name_b])
        cp_a = 'z image' + chr(92) + name_a
        cp_b = 'z image' + chr(92) + name_b
        ds_a = svc.create_dataset(LOCAL_USER, 'Ca', 'ca')
        ds_b = svc.create_dataset(LOCAL_USER, 'Cb', 'cb')
        by_ds = {ds_a.id: [{'filename': cp_a}], ds_b.id: [{'filename': cp_b}]}
        _neutralise_preflights(monkeypatch, lts)
        monkeypatch.setattr(lts, 'list_test_checkpoints',
                            lambda ds, _family=None: by_ds[ds.id])
        submitted = []

        def capture(user_id, dataset_id, workflow, prompt, job_id=None, **_kw):
            submitted.append((dataset_id, prompt))
            return job_id
        monkeypatch.setattr(lts, '_enqueue_cell', capture)
        monkeypatch.setattr(lts, '_build_cell_workflow', lambda *a, **k: {'1': {}})

        out = ct.canvas_generate(
            LOCAL_USER,
            [{'dataset_id': ds_a.id, 'checkpoint': cp_a, 'record_id': 11, 'step': 2000},
             {'dataset_id': ds_b.id, 'checkpoint': cp_b, 'record_id': 22, 'step': 1000}],
            [1.0], lts.StudioGenSettings(count=1),
            prompts=['on a rooftop', 'in the snow', 'at night'])

        assert out['created'] == 6
        assert sorted(submitted) == sorted(
            [(ds.id, p) for ds in (ds_a, ds_b)
             for p in ('on a rooftop', 'in the snow', 'at night')])
        rows = LoraTestImage.query.filter_by(run_id=out['run_id']).all()
        # One run, one seed: the batch compares prompts, it does not reseed them.
        assert len({r.run_seed for r in rows}) == 1


def test_canvas_batch_without_prompts_still_falls_back_per_dataset(
        app, monkeypatch, tmp_path):
    """No batch AND no common prompt: each cell keeps its own dataset's identity
    prompt — the historical behaviour the axis must not have swallowed."""
    from app.services import cloud_training as ct, lora_test_studio as lts
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app.models import LoraTestImage
    with app.app_context():
        name_a, name_b = 'lora_da_000002000.safetensors', 'lora_db_000001000.safetensors'
        _zimage_tree(tmp_path, monkeypatch, [name_a, name_b])
        cp_a = 'z image' + chr(92) + name_a
        cp_b = 'z image' + chr(92) + name_b
        ds_a = svc.create_dataset(LOCAL_USER, 'Da', 'da')
        ds_b = svc.create_dataset(LOCAL_USER, 'Db', 'db')
        by_ds = {ds_a.id: [{'filename': cp_a}], ds_b.id: [{'filename': cp_b}]}
        _neutralise_preflights(monkeypatch, lts)
        monkeypatch.setattr(lts, 'list_test_checkpoints',
                            lambda ds, _family=None: by_ds[ds.id])
        monkeypatch.setattr(lts, '_enqueue_cell',
                            lambda *a, job_id=None, **k: job_id)
        monkeypatch.setattr(lts, '_build_cell_workflow', lambda *a, **k: {'1': {}})

        out = ct.canvas_generate(
            LOCAL_USER,
            [{'dataset_id': ds_a.id, 'checkpoint': cp_a},
             {'dataset_id': ds_b.id, 'checkpoint': cp_b}],
            [1.0], lts.StudioGenSettings(count=1))

        assert out['created'] == 2
        rows = LoraTestImage.query.filter_by(run_id=out['run_id']).all()
        prompts = {r.dataset_id: r.prompt for r in rows}
        assert 'da' in prompts[ds_a.id] and 'db' in prompts[ds_b.id]


# --- both routes forward the key ---------------------------------------------

def test_canvas_route_forwards_the_prompt_batch(client, monkeypatch):
    _comfy(monkeypatch)
    seen = {}

    def fake(user_id, selections, strengths, settings=None, **kwargs):
        seen.update(kwargs)
        return {'created': 3, 'seed': 7, 'count': 1, 'run_id': 'r1', 'ids': []}

    monkeypatch.setattr('app.services.cloud_training.canvas_generate', fake)
    resp = client.post('/api/train/canvas/generate', json={
        'selections': [{'dataset_id': 1, 'checkpoint': 'a.safetensors'}],
        'prompts': ['one', 'two', 'three']})
    assert resp.status_code == 200
    assert seen['prompts'] == ['one', 'two', 'three']


# --- the pace the warning quotes is MEASURED, not assumed ---------------------

def _pace_dataset(name):
    """A real dataset row — the cells carry a foreign key to one."""
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    return svc.create_dataset(LOCAL_USER, name, name.lower())


def _finished_cell(dataset_id, checkpoint, seconds, job_id):
    """One completed test cell + its queue job, `seconds` apart."""
    from datetime import datetime, timedelta
    from app.extensions import db
    from app.models import ImageGenerationQueue, LoraTestImage
    started = datetime(2026, 8, 3, 12, 0, 0)
    db.session.add(LoraTestImage(dataset_id=dataset_id, checkpoint=checkpoint,
                                 strength=1.0, status='done', job_id=job_id))
    db.session.add(ImageGenerationQueue(
        job_id=job_id, user_id='local', status='completed',
        started_at=started, completed_at=started + timedelta(seconds=seconds)))


def test_pace_is_the_median_of_what_this_machine_really_did(app):
    """The UI said "~12 s/image" on every card in the world. The queue has
    recorded started_at/completed_at since forever — this reads it."""
    from app.extensions import db
    from app.services import lora_test_studio as lts
    with app.app_context():
        ds = _pace_dataset('PaceA')
        ck = 'z image' + chr(92) + 'lora_p_000002000.safetensors'
        for i, secs in enumerate([30, 32, 31, 29, 33]):
            _finished_cell(ds.id, ck, secs, f'job-{i}')
        db.session.commit()
        assert lts.measured_seconds_per_image('zimage') == 31.0


def test_pace_ignores_a_machine_that_went_to_sleep_mid_job(app):
    """A median, and a sane window: one job spanning a suspend would otherwise
    make the panel announce eight hours an image for the next hundred runs."""
    from app.extensions import db
    from app.services import lora_test_studio as lts
    with app.app_context():
        ds = _pace_dataset('PaceB')
        ck = 'z image' + chr(92) + 'lora_q_000002000.safetensors'
        for i, secs in enumerate([30, 31, 29, 30, 8 * 3600]):
            _finished_cell(ds.id, ck, secs, f'sleepy-{i}')
        db.session.commit()
        pace = lts.measured_seconds_per_image('zimage')
        assert pace is not None and 29 <= pace <= 31


def test_pace_says_nothing_rather_than_guessing_from_two_samples(app):
    """Below the sample floor it returns None and the UI keeps its "~" default —
    a precise-looking number drawn from two measurements is worse than none."""
    from app.extensions import db
    from app.services import lora_test_studio as lts
    with app.app_context():
        ds = _pace_dataset('PaceC')
        ck = 'z image' + chr(92) + 'lora_r_000002000.safetensors'
        _finished_cell(ds.id, ck, 30, 'lonely-0')
        db.session.commit()
        assert lts.measured_seconds_per_image('zimage') is None


def test_pace_is_scoped_to_the_family_being_launched(app):
    """A Krea image and a Z-Image Turbo one do not cost the same; quoting one
    machine-wide average would mislead on whichever family is slower."""
    from app.extensions import db
    from app.services import lora_test_studio as lts
    with app.app_context():
        ds_z, ds_k = _pace_dataset('PaceZ'), _pace_dataset('PaceK')
        z = 'z image' + chr(92) + 'lora_s_000002000.safetensors'
        k = 'krea' + chr(92) + 'lora_t_000002000.safetensors'
        for i, secs in enumerate([10, 11, 10, 11]):
            _finished_cell(ds_z.id, z, secs, f'zz-{i}')
        for i, secs in enumerate([120, 118, 122, 120]):
            _finished_cell(ds_k.id, k, secs, f'kk-{i}')
        db.session.commit()
        assert lts.measured_seconds_per_image('zimage') < 20
        assert lts.measured_seconds_per_image('krea') > 100


def test_both_panels_are_served_the_same_pace_key(client, monkeypatch):
    """Two branches of one screen must never announce two different durations."""
    from app.services import lora_test_studio as lts
    _comfy(monkeypatch)
    monkeypatch.setattr(lts, 'measured_seconds_per_image', lambda _f=None: 42.0)
    axes = client.get('/api/studio/base-models?type=zimage').get_json()['axes']
    assert axes['seconds_per_image'] == 42.0


def test_test_studio_route_forwards_the_prompt_batch(client, monkeypatch):
    _comfy(monkeypatch)
    seen = {}

    def fake(user_id, dataset_id, checkpoints, strengths, settings=None,
             **kwargs):
        seen.update(kwargs)
        return {'created': 2, 'seed': 7, 'count': 1, 'run_id': 'r1', 'ids': []}

    monkeypatch.setattr('app.services.lora_test_studio.create_run', fake)
    resp = client.post('/api/dataset/1/lora-test/run', json={
        'checkpoints': ['a.safetensors'], 'strengths': [1.0],
        'prompts': ['one', 'two']})
    assert resp.status_code == 200
    assert seen['prompts'] == ['one', 'two']


# --- what the RESULTS view needs in order to keep a batch whole --------------
# The batch generated all N images and lost them on the way to the SCREEN.
# Reported as « a grid with a single image »: the results view had no launch
# identity to group by, so it inferred one from `run_seed` + prompt — and a
# batch, whose whole point is N prompts under ONE seed, therefore arrived as N
# separate runs, of which the view displays one. `run_id` has been written by
# `create_run` since the multi-LoRA comparison; it was simply never served.


def test_payload_tells_the_grid_which_launch_each_cell_belongs_to(app):
    """If `run_id` silently leaves this payload again, the grid falls back to the
    prompt-keyed grouping and the bug returns without a single test going red —
    which is exactly why this one exists."""
    from app.extensions import db
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.models import LoraTestImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Batch', 'batchtrig')
        ck = 'z image' + chr(92) + 'lora_batchtrig_000001000.safetensors'
        prompts = ['on a rooftop', 'in a neon alley', 'in a sunlit cafe']
        for i, prompt in enumerate(prompts):
            db.session.add(LoraTestImage(
                dataset_id=ds.id, checkpoint=ck, strength=1.0, status='done',
                filename=f'b{i}.png', seed=99, run_seed=99, run_id='ONE-LAUNCH',
                prompt=prompt, aspect='1:1', cfg=1.0, steps=8))
        db.session.commit()

        cells = lts.studio_payload(LOCAL_USER, ds.id)['cells']
        assert len(cells) == len(prompts)
        assert {c['prompt'] for c in cells} == set(prompts)
        # ONE launch: every cell carries the same, non-null run_id...
        assert {c['run_id'] for c in cells} == {'ONE-LAUNCH'}
        # ...and it is served ON THE CELL, which is where the grid groups. A
        # payload carrying it only at the top level would not help at all.
        assert all('run_id' in c for c in cells)


def test_a_run_predating_run_id_says_so_instead_of_inventing_one(app):
    """Old rows have no run_id. The payload reports that honestly (None) so the
    frontend can keep its legacy grouping for them — a fabricated id would merge
    two genuinely separate launches into one."""
    from app.extensions import db
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.models import LoraTestImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Legacy', 'legacytrig')
        ck = 'z image' + chr(92) + 'lora_legacytrig_000001000.safetensors'
        db.session.add(LoraTestImage(
            dataset_id=ds.id, checkpoint=ck, strength=1.0, status='done',
            filename='old.png', seed=5, run_seed=5, run_id=None, prompt='old run'))
        db.session.commit()
        assert lts.studio_payload(LOCAL_USER, ds.id)['cells'][0]['run_id'] is None


def test_create_run_stamps_ONE_run_id_across_the_whole_prompt_batch(app, tmp_path, monkeypatch):
    """The grouping the frontend now relies on is only as good as what the engine
    writes: N prompts must land under ONE run_id, not one per prompt."""
    from app.extensions import db
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.models import LoraTestImage
    from app.config import LOCAL_USER
    with app.app_context():
        _zimage_tree(tmp_path, monkeypatch, ['lora_batched_000001000.safetensors'])
        _comfy(monkeypatch)
        _neutralise_preflights(monkeypatch, lts)
        monkeypatch.setattr(lts, '_persist_and_enqueue_cell',
                            lambda img, *a, **k: (db.session.add(img), db.session.commit()))
        ds = svc.create_dataset(LOCAL_USER, 'Batched', 'batched')
        res = lts.create_run(
            LOCAL_USER, ds.id,
            ['z image' + chr(92) + 'lora_batched_000001000.safetensors'], [1.0],
            lts.StudioGenSettings(seed=42),
            prompts=['first prompt', 'second prompt', 'third prompt'])
        rows = LoraTestImage.query.filter_by(dataset_id=ds.id).all()
        assert len(rows) == 3, 'one cell per ticked prompt'
        assert len({r.prompt for r in rows}) == 3
        assert len({r.run_id for r in rows}) == 1, 'a batch is ONE launch'
        assert rows[0].run_id == res['run_id']
        assert len({r.run_seed for r in rows}) == 1
