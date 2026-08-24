"""The FINAL save of a run carries no step number in its filename
(`lora_<trigger>.safetensors`, never `..._000003000.safetensors`). Two views used
to number it differently: the ◉ Graph, reading a cloud run's staging, called it
the run's target (3000), while the local checkpoint list — the "active set" the
▶ Continue dialog resumes — filed it under the last NUMBERED save (2750), where
the dialog's dedup swallowed it. A pill the graph offered was then refused with a
message blaming the family/base/variant selection.

Reported by the owner on a Krea 2 run: 12 saves on disk, 250…2750 numbered plus
the unnumbered final, run declared 3000 steps, pill "3k" not resumable.
"""
import datetime
from app.utils.timestamps import naive_utcnow
import os


def _configure_aitoolkit(tmp_path):
    """Minimal ai-toolkit install so the run dir resolves (no training is run)."""
    from app import config as cfg
    root = tmp_path / 'aitoolkit'
    (root / 'venv' / 'Scripts').mkdir(parents=True)
    (root / 'venv' / 'Scripts' / 'python.exe').write_text('fake')
    (root / 'run.py').write_text('fake')
    cfg.save_config({'aitoolkit': {'dir': str(root)}})


def _seed(lt, svc, LOCAL_USER, tmp_path, numbered, target_steps, final=True):
    """A krea/base dataset whose run dir holds `numbered` saves + the unnumbered
    final, registered to a local run record of `target_steps` steps.

    The record is LOCAL on purpose. list_checkpoints resolves provenance with
    `source=('local', 'legacy')` so an unrelated cloud launch cannot steal a
    local checkpoint's version and lineage by mtime alone; a cloud record
    therefore no longer numbers a file in the local run dir. Cloud saves are
    listed by the route's separate cloud_checkpoint_groups field."""
    from app.extensions import db
    from app.models import TrainingRunRecord
    _configure_aitoolkit(tmp_path)
    ds = svc.create_dataset(LOCAL_USER, 'Final step', 'finalstep')
    ds.train_type = 'krea'
    ds.train_variant = 'base'
    db.session.commit()
    trigger = lt._safe_trigger(ds)
    run_dir = lt._run_dir(LOCAL_USER, ds.id, None, 'krea', 'base')
    os.makedirs(run_dir, exist_ok=True)
    for s in numbered:
        with open(os.path.join(run_dir, f'lora_{trigger}_{s:09d}.safetensors'), 'wb') as fh:
            fh.write(f'W{s}'.encode())
    if final:
        with open(os.path.join(run_dir, f'lora_{trigger}.safetensors'), 'wb') as fh:
            fh.write(b'FINAL')
    db.session.add(TrainingRunRecord(
        dataset_id=ds.id, family='krea', source='local',
        base_model='', variant='base', steps=target_steps, version=1,
        fingerprint='fp', manifest='[]',
        created_at=naive_utcnow() - datetime.timedelta(days=1)))
    db.session.commit()
    return ds, run_dir


def test_unnumbered_final_is_listed_at_its_run_target_step(app, tmp_path):
    """The 3000-step end of the run is listed AT 3000 — the number the graph
    pill shows — instead of hiding behind a second 2750 entry."""
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    with app.app_context():
        ds, _ = _seed(lt, svc, LOCAL_USER, tmp_path,
                      [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750],
                      3000)
        cks = lt.list_checkpoints(LOCAL_USER, ds.id, None, 'krea', 'base')
        assert len(cks) == 12                      # 11 numbered + the final
        final = [c for c in cks if c.get('final')]
        assert len(final) == 1
        assert final[0]['step'] == 3000
        # ascending, final last: the dialog's "latest" is the run's real end
        assert [c['step'] for c in cks][-1] == 3000
        assert sorted({c['step'] for c in cks}) == [
            250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000]


def test_that_final_step_is_resumable(app, tmp_path):
    """And it can actually be continued FROM: the graph pill's step resolves to a
    real file instead of 'no local checkpoint at step 3000'."""
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    with app.app_context():
        ds, _ = _seed(lt, svc, LOCAL_USER, tmp_path, [2500, 2750], 3000)
        path = lt.checkpoint_file_path(
            LOCAL_USER, ds.id,
            [c for c in lt.list_checkpoints(LOCAL_USER, ds.id, None, 'krea', 'base')
             if c['step'] == 3000][0]['filename'],
            None, 'krea', 'base')
        assert path and os.path.isfile(path)


def test_no_run_record_keeps_the_historical_numbering(app, tmp_path):
    """Nothing to learn from (pre-registry files, no record): the final save
    keeps its historical step — never an invented number."""
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    from app.extensions import db
    from app.models import TrainingRunRecord
    with app.app_context():
        ds, _ = _seed(lt, svc, LOCAL_USER, tmp_path, [500, 1000], 1500)
        TrainingRunRecord.query.filter_by(dataset_id=ds.id).delete()
        db.session.commit()
        cks = lt.list_checkpoints(LOCAL_USER, ds.id, None, 'krea', 'base')
        assert [c for c in cks if c.get('final')][0]['step'] == 1000


def test_a_run_that_stopped_short_never_moves_backwards(app, tmp_path):
    """A record whose target is BELOW the last numbered save (an early-stopped
    run relaunched shorter) must not renumber the final downwards."""
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    with app.app_context():
        ds, _ = _seed(lt, svc, LOCAL_USER, tmp_path, [500, 1000], 800)
        cks = lt.list_checkpoints(LOCAL_USER, ds.id, None, 'krea', 'base')
        assert [c for c in cks if c.get('final')][0]['step'] == 1000
