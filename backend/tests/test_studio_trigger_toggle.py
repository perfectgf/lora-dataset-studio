"""Case « Trigger word » — the Studio checkbox that sends the prompt as written.

Every Studio launch used to prefix the dataset's trigger word to the test
prompt at workflow-build time, unconditionally. That is the right default for
identity LoRAs, but it leaks the trigger into renders that TYPE text (a speech
bubble asked to "say hello" writes the trigger word inside the bubble), and a
style/scene test sometimes wants the prompt verbatim. The checkbox travels as
`inject_trigger` on the run payload: absent/None/True = the historical
behaviour, byte for byte; False = the workflow builder receives NO trigger
word, the cell row remembers the choice (`LoraTestImage.inject_trigger` False,
NULL otherwise) and resume honours what the row says.

Pinned here:
  · a default launch keeps passing the dataset trigger to the builder and
    leaves the column NULL — new default rows stay indistinguishable from
    legacy rows, on purpose;
  · `inject_trigger=False` reaches BOTH engines (`create_run` for the Test
    Studio, `create_comparison_run` for compare/canvas) as trigger_word=None
    and stamps False on every row;
  · resume rebuilds a False cell without a trigger and a NULL (legacy) cell
    with one — "resumable with their settings" includes this setting;
  · the wire name is read by `StudioGenSettings.from_payload`;
  · the run payload serves the column so the results meta can say it.
"""


def _prep(monkeypatch, lts, checkpoints):
    """Krea-family run with externals removed; returns the captured
    (trigger_word passed to the builder, enqueued job prompts) lists."""
    triggers, prompts = [], []
    monkeypatch.setattr(lts, 'gpu_busy_reason', lambda: None)
    monkeypatch.setattr(lts, '_active_run_count', lambda *a: 0)
    monkeypatch.setattr(
        lts, 'list_test_checkpoints',
        lambda _ds, _family=None: [{'filename': c} for c in checkpoints])
    monkeypatch.setattr(lts, 'permanent_lora_candidates', lambda _f: [])
    monkeypatch.setattr(lts, 'get_krea_models', lambda: [])
    monkeypatch.setattr(lts, '_preflight_checkpoint_arch', lambda *a, **k: None)
    monkeypatch.setattr(lts, '_preflight_run', lambda *a, **k: None)
    monkeypatch.setattr(lts, '_target_node_classes', lambda: None)
    monkeypatch.setattr(
        lts, 'checkpoint_origins',
        lambda selected, explicit=None: {c: (None, None) for c in selected})

    def fake_builder(*a, **k):
        triggers.append(k.get('trigger_word'))
        return {'1': {'prompt': a[3]}}
    monkeypatch.setattr(lts, '_build_cell_workflow', fake_builder)

    def fake_enqueue(user_id, dataset_id, workflow, prompt, job_id=None, **_k):
        prompts.append(prompt)
        return job_id
    monkeypatch.setattr(lts, '_enqueue_cell', fake_enqueue)
    return triggers, prompts


CK = 'krea\\lora_ktog_000001000.safetensors'


def test_default_launch_still_injects_and_leaves_the_column_null(
        app, monkeypatch):
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.config import LOCAL_USER
    from app.models import LoraTestImage
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Toggle default', 'ktog')
        triggers, _ = _prep(monkeypatch, lts, [CK])
        out = lts.create_run(LOCAL_USER, ds.id, [CK], [1.0],
                             lts.StudioGenSettings(prompt='p', count=1))
        assert triggers == [ds.trigger_word]
        row = LoraTestImage.query.filter_by(run_id=out['run_id']).one()
        assert row.inject_trigger is None


def test_unchecked_box_reaches_the_builder_as_no_trigger_and_marks_the_row(
        app, monkeypatch):
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.config import LOCAL_USER
    from app.models import LoraTestImage
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Toggle off', 'ktog')
        triggers, _ = _prep(monkeypatch, lts, [CK])
        out = lts.create_run(
            LOCAL_USER, ds.id, [CK], [1.0],
            lts.StudioGenSettings(prompt='p', count=1, inject_trigger=False))
        assert triggers == [None]
        row = LoraTestImage.query.filter_by(run_id=out['run_id']).one()
        assert row.inject_trigger is False
        # The run payload serves the choice — the results meta can say it.
        payload = lts.studio_payload_run(LOCAL_USER, out['run_id'])
        assert payload['cells'][0]['inject_trigger'] is False


def test_comparison_engine_honours_the_box_on_every_cell(app, monkeypatch):
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.config import LOCAL_USER
    from app.models import LoraTestImage
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Toggle compare', 'ktog')
        triggers, _ = _prep(monkeypatch, lts, [CK])
        out = lts.create_comparison_run(
            LOCAL_USER, [{'dataset_id': ds.id, 'checkpoint': CK}], [1.0, 0.5],
            lts.StudioGenSettings(prompt='p', count=1, inject_trigger=False))
        assert triggers == [None, None]
        rows = LoraTestImage.query.filter_by(run_id=out['run_id']).all()
        assert [r.inject_trigger for r in rows] == [False, False]


def test_resume_honours_what_each_row_recorded(app, monkeypatch):
    """A False cell resumes without a trigger; a NULL (legacy) cell keeps it."""
    from app import db
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.config import LOCAL_USER
    from app.models import LoraTestImage
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Toggle resume', 'ktog')
        triggers, _ = _prep(monkeypatch, lts, [CK])
        for flag in (False, None):
            db.session.add(LoraTestImage(
                dataset_id=ds.id, checkpoint=CK, strength=1.0, seed=7,
                run_id='run-toggle', status='failed', prompt='p',
                inject_trigger=flag))
        db.session.commit()
        out = lts.resume_run(LOCAL_USER, run_id='run-toggle')
        assert out['resumed'] == 2
        assert sorted(triggers, key=repr) == sorted(
            [None, ds.trigger_word], key=repr)


def test_wire_name_is_read_by_from_payload():
    from app.services import lora_test_studio as lts
    assert lts.StudioGenSettings.from_payload({}).inject_trigger is None
    assert lts.StudioGenSettings.from_payload(
        {'inject_trigger': False}).inject_trigger is False
    assert lts.StudioGenSettings.from_payload(
        {'inject_trigger': True}).inject_trigger is True
