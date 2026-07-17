"""Targeted regression tests for SQLite integrity and user-data Trash paths."""
import json
import os
import sqlite3

import pytest
from sqlalchemy import text


def test_sqlite_connections_enforce_foreign_keys_and_new_schema_cascades(app):
    from app.extensions import db
    from app.models import FaceDataset, LoraTestImage

    with app.app_context():
        with db.engine.connect() as connection:
            assert connection.execute(text('PRAGMA foreign_keys')).scalar_one() == 1

        fk_rows = db.session.execute(
            text('PRAGMA foreign_key_list(lora_test_image)')).all()
        assert any(row[2] == 'face_dataset' and row[6].upper() == 'CASCADE'
                   for row in fk_rows)

        ds = FaceDataset(user_id='local', name='Cascade', trigger_word='cascade')
        db.session.add(ds)
        db.session.flush()
        cell = LoraTestImage(dataset_id=ds.id, checkpoint='lora.safetensors',
                             strength=1.0)
        db.session.add(cell)
        db.session.commit()
        dataset_id = ds.id

        db.session.execute(
            text('DELETE FROM face_dataset WHERE id = :dataset_id'),
            {'dataset_id': dataset_id})
        db.session.commit()
        assert LoraTestImage.query.filter_by(dataset_id=dataset_id).count() == 0


def test_startup_cleans_only_legacy_orphaned_studio_rows(tmp_path, monkeypatch):
    """Clean legacy cells; cancel only queue jobs with an exact Studio linkage."""
    data_dir = tmp_path / 'legacy-data'
    data_dir.mkdir()
    monkeypatch.setenv('LDS_DATA_DIR', str(data_dir))
    monkeypatch.setenv('LDS_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setenv('LDS_ENV', str(tmp_path / '.env'))
    from app import config as cfg
    monkeypatch.setattr(cfg, 'ENV_PATH', tmp_path / '.env')
    monkeypatch.setattr(cfg, '_cache', None)
    from app import create_app
    from app.extensions import db
    from app.models import FaceDataset

    first_boot = create_app({
        'TESTING': True,
        'WTF_CSRF_ENABLED': False,
    })
    with first_boot.app_context():
        db.session.add(FaceDataset(
            id=1, user_id='local', name='Valid', trigger_word='valid'))
        db.session.commit()
        db.session.remove()
        db.engine.dispose()

    # Raw SQLite defaults to FK=OFF, reproducing a database written by the old
    # app. Its full schema came from first_boot, so queue metadata can be tested.
    db_path = data_dir / 'studio.db'
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            'INSERT INTO lora_test_image '
            '(id, dataset_id, checkpoint, strength, status, rating) '
            "VALUES (10, 1, 'valid.safetensors', 1.0, 'done', 0)")
        connection.execute(
            'INSERT INTO lora_test_image '
            '(id, dataset_id, checkpoint, strength, status, rating, job_id) '
            "VALUES (11, 999, 'orphan.safetensors', 1.0, 'pending', 0, 'safe-job')")
        connection.execute(
            'INSERT INTO lora_test_image '
            '(id, dataset_id, checkpoint, strength, status, rating, job_id) '
            "VALUES (12, 998, 'orphan.safetensors', 1.0, 'pending', 0, 'unsafe-job')")
        safe_metadata = json.dumps({
            'model_name': 'zimage_lora_test', 'is_lora_test': True,
            'dataset_id': 999})
        unsafe_metadata = json.dumps({
            'model_name': 'zimage_lora_test', 'is_lora_test': True,
            'dataset_id': 123})
        for job_id, metadata in (
                ('safe-job', safe_metadata), ('unsafe-job', unsafe_metadata)):
            connection.execute(
                'INSERT INTO image_generation_queue '
                '(job_id, user_id, status, retry_count, priority, created_at, '
                'job_metadata) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)',
                (job_id, 'local', 'pending', 0, 0, metadata))
        connection.commit()
    finally:
        connection.close()

    application = create_app({
        'TESTING': True,
        'WTF_CSRF_ENABLED': False,
    })
    with application.app_context():
        rows = db.session.execute(
            text('SELECT id, dataset_id FROM lora_test_image ORDER BY id')).all()
        assert rows == [(10, 1)]
        statuses = dict(db.session.execute(text(
            'SELECT job_id, status FROM image_generation_queue')).all())
        assert statuses == {'safe-job': 'cancelled', 'unsafe-job': 'pending'}
        assert db.session.execute(text('PRAGMA foreign_keys')).scalar_one() == 1


def _trash_spy(monkeypatch):
    from app.services import trash

    calls = []
    original = trash.send_to_trash

    def spy(path, context=''):
        destination = original(path, context=context)
        calls.append((str(path), context, destination))
        return destination

    monkeypatch.setattr(trash, 'send_to_trash', spy)
    return calls


def test_delete_image_rolls_back_queue_cancellation_when_commit_fails(
        app, monkeypatch):
    from app.extensions import db
    from app.job_queue import queue_manager
    from app.models import FaceDatasetImage, ImageGenerationQueue
    from app.services import face_dataset_service as svc

    with app.app_context():
        ds = svc.create_dataset('local', 'Atomic image', 'atomic_image')
        job_id = queue_manager.add_job(
            workflow_data={'1': {}}, user_id='local',
            metadata={'is_dataset': True, 'dataset_id': ds.id})
        image = FaceDatasetImage(
            dataset_id=ds.id, source='generated', status='pending',
            filename=None, job_id=job_id)
        db.session.add(image)
        db.session.commit()
        image_id = image.id

        session = db.session()

        def fail_commit():
            raise RuntimeError('injected delete commit failure')

        monkeypatch.setattr(session, 'commit', fail_commit)
        with pytest.raises(RuntimeError, match='injected delete commit failure'):
            svc.delete_image('local', image_id)

        assert db.session.get(FaceDatasetImage, image_id) is not None
        assert (ImageGenerationQueue.query.filter_by(job_id=job_id).one().status
                == 'pending')


def test_studio_prompt_delete_trashes_files_and_cancels_jobs_atomically(
        app, monkeypatch):
    from app.extensions import db
    from app.job_queue import queue_manager
    from app.models import ImageGenerationQueue, LoraTestImage
    from app.services import face_dataset_service as fds
    from app.services import lora_test_studio as studio

    with app.app_context():
        calls = _trash_spy(monkeypatch)
        ds = fds.create_dataset('local', 'Studio trash', 'studio_trash')
        folder = fds._dataset_dir(ds.id)
        result_path = os.path.join(folder, 'cell.png')
        with open(result_path, 'wb') as fh:
            fh.write(b'cell')
        job_id = queue_manager.add_job(
            workflow_data={'1': {}}, user_id='local',
            metadata={'model_name': 'zimage_lora_test', 'is_lora_test': True,
                      'dataset_id': ds.id})
        db.session.add_all((
            LoraTestImage(
                dataset_id=ds.id, checkpoint='done.safetensors', strength=1.0,
                prompt='delete me', status='done', filename='cell.png'),
            LoraTestImage(
                dataset_id=ds.id, checkpoint='pending.safetensors', strength=1.0,
                prompt='delete me', status='pending', job_id=job_id),
        ))
        db.session.commit()

        assert studio.delete_prompt('local', ds.id, 'delete me') == 2
        assert LoraTestImage.query.filter_by(dataset_id=ds.id).count() == 0
        assert not os.path.exists(result_path)
        assert (ImageGenerationQueue.query.filter_by(job_id=job_id).one().status
                == 'cancelled')
        assert len(calls) == 1
        assert calls[0][1] == f'dataset-{ds.id}-studio-prompt'
        assert os.path.exists(calls[0][2])


def test_studio_prompt_delete_restores_file_and_rows_when_commit_fails(
        app, monkeypatch):
    from app.extensions import db
    from app.models import LoraTestImage
    from app.services import face_dataset_service as fds
    from app.services import lora_test_studio as studio

    with app.app_context():
        calls = _trash_spy(monkeypatch)
        ds = fds.create_dataset('local', 'Studio rollback', 'studio_rollback')
        folder = fds._dataset_dir(ds.id)
        result_path = os.path.join(folder, 'cell.png')
        with open(result_path, 'wb') as fh:
            fh.write(b'cell')
        db.session.add(LoraTestImage(
            dataset_id=ds.id, checkpoint='done.safetensors', strength=1.0,
            prompt='keep me', status='done', filename='cell.png'))
        db.session.commit()

        session = db.session()

        def fail_commit():
            raise RuntimeError('injected prompt commit failure')

        monkeypatch.setattr(session, 'commit', fail_commit)
        with pytest.raises(RuntimeError, match='injected prompt commit failure'):
            studio.delete_prompt('local', ds.id, 'keep me')

        assert LoraTestImage.query.filter_by(dataset_id=ds.id).count() == 1
        assert os.path.isfile(result_path)
        assert len(calls) == 1
        assert not os.path.exists(calls[0][2])


def test_regenerate_preflight_failure_keeps_current_file_out_of_trash(
        app, monkeypatch):
    from app.extensions import db
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.services import trash

    with app.app_context():
        ds = svc.create_dataset('local', 'Preflight', 'preflight')
        folder = svc._dataset_dir(ds.id)
        old_path = os.path.join(folder, 'old.webp')
        with open(old_path, 'wb') as fh:
            fh.write(b'old')
        ds.ref_filename = 'missing-ref.webp'
        image = FaceDatasetImage(
            dataset_id=ds.id, source='generated', status='keep',
            filename='old.webp', variation_prompt='portrait')
        db.session.add(image)
        db.session.commit()
        image_id = image.id
        calls = []
        monkeypatch.setattr(svc, '_api_generate_fn', lambda _engine: lambda *_a, **_k: b'new')
        monkeypatch.setattr(
            trash, 'send_to_trash',
            lambda *args, **kwargs: calls.append((args, kwargs)))

        with pytest.raises(ValueError, match='reference image file missing'):
            svc.regenerate_image('local', image_id, engine='nanobanana')

        row = db.session.get(FaceDatasetImage, image_id)
        assert row.filename == 'old.webp' and row.status == 'keep'
        assert os.path.isfile(old_path)
        assert calls == []


def test_regenerate_trash_failure_restores_previous_row_and_cancels_new_job(
        app, monkeypatch):
    from app.extensions import db
    from app.job_queue import queue_manager
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper, trash

    with app.app_context():
        ds = svc.create_dataset('local', 'Trash rollback', 'trash_rollback')
        folder = svc._dataset_dir(ds.id)
        ref_path = os.path.join(folder, 'ref.webp')
        old_path = os.path.join(folder, 'old.webp')
        for path, content in ((ref_path, b'ref'), (old_path, b'old')):
            with open(path, 'wb') as fh:
                fh.write(content)
        ds.ref_filename = 'ref.webp'
        image = FaceDatasetImage(
            dataset_id=ds.id, source='generated', status='keep',
            filename='old.webp', caption='old caption',
            variation_prompt='portrait', klein_model='klein.safetensors',
            watermark_state='detected', watermark_bbox='[0,0,1,1]')
        db.session.add(image)
        db.session.commit()
        image_id = image.id
        monkeypatch.setattr(
            klein_edit_helper, 'enqueue_klein_edit', lambda **_kwargs: 'new-job')
        cancellations = []

        def cancel(job_id, user_id=None, job_type='image', *, commit=True):
            cancellations.append((job_id, user_id, job_type, commit))
            return True

        monkeypatch.setattr(queue_manager, 'cancel_job', cancel)

        def fail_trash(_path, context=''):
            db.session.expire_all()
            pending = db.session.get(FaceDatasetImage, image_id)
            assert pending.filename is None and pending.status == 'pending'
            assert pending.job_id == 'new-job'
            raise OSError('injected Trash failure')

        monkeypatch.setattr(trash, 'send_to_trash', fail_trash)
        with pytest.raises(OSError, match='injected Trash failure'):
            svc.regenerate_image('local', image_id, engine='klein')

        row = db.session.get(FaceDatasetImage, image_id)
        assert (row.filename, row.status, row.caption, row.job_id) == (
            'old.webp', 'keep', 'old caption', None)
        assert row.watermark_state == 'detected'
        assert os.path.isfile(old_path)
        assert ('new-job', 'local', 'image', False) in cancellations


def test_dataset_deletions_are_owned_trashed_and_leave_no_studio_orphans(
        app, monkeypatch):
    from app.extensions import db
    from app.job_queue import queue_manager
    from app.models import FaceDatasetImage, ImageGenerationQueue, LoraTestImage
    from app.services import face_dataset_service as svc

    with app.app_context():
        calls = _trash_spy(monkeypatch)
        ds = svc.create_dataset('local', 'Trash safety', 'trashsafe')
        folder = svc._dataset_dir(ds.id)

        image_path = os.path.join(folder, 'image.webp')
        with open(image_path, 'wb') as fh:
            fh.write(b'image')
        image = FaceDatasetImage(dataset_id=ds.id, filename='image.webp',
                                 status='keep', source='import')
        db.session.add(image)

        extra_path = os.path.join(folder, 'extra.webp')
        with open(extra_path, 'wb') as fh:
            fh.write(b'extra')
        ds.ref_extra_filenames = json.dumps(['extra.webp'])

        ref_path = os.path.join(folder, 'ref.webp')
        with open(ref_path, 'wb') as fh:
            fh.write(b'reference')
        ds.ref_filename = 'ref.webp'
        studio_job = queue_manager.add_job(
            workflow_data={'1': {}}, user_id='local',
            metadata={'model_name': 'zimage_lora_test', 'is_lora_test': True,
                      'dataset_id': ds.id})
        face_job = queue_manager.add_job(
            workflow_data={'1': {}}, user_id='local',
            metadata={'model_name': 'klein_edit_dataset', 'is_dataset': True,
                      'dataset_id': ds.id})
        cell = LoraTestImage(dataset_id=ds.id, checkpoint='lora.safetensors',
                             strength=1.0, status='pending', job_id=studio_job)
        pending_image = FaceDatasetImage(
            dataset_id=ds.id, filename=None, status='pending',
            source='generated', job_id=face_job)
        db.session.add_all((cell, pending_image))
        db.session.commit()
        image_id, dataset_id = image.id, ds.id

        assert svc.delete_image('local', image_id) is True
        assert not os.path.exists(image_path)
        assert svc.remove_extra_ref('local', dataset_id, 'extra.webp') is True
        assert not os.path.exists(extra_path)

        # Emulate an old DB where the FK existed without enforcement/cascade:
        # service-level explicit cleanup must still remove Studio rows.
        db.session.execute(text('PRAGMA foreign_keys=OFF'))
        assert db.session.execute(text('PRAGMA foreign_keys')).scalar_one() == 0
        db.session.commit()

        before_foreign_attempt = len(calls)
        assert svc.delete_dataset('another-user', dataset_id) is False
        assert len(calls) == before_foreign_attempt
        assert os.path.isdir(folder)
        assert LoraTestImage.query.filter_by(dataset_id=dataset_id).count() == 1

        assert svc.delete_dataset('local', dataset_id) is True
        assert not os.path.exists(folder)
        assert LoraTestImage.query.filter_by(dataset_id=dataset_id).count() == 0
        queue_states = dict(
            db.session.query(ImageGenerationQueue.job_id,
                             ImageGenerationQueue.status).all())
        assert queue_states[studio_job] == 'cancelled'
        assert queue_states[face_job] == 'cancelled'
        assert {context for _src, context, _dst in calls} >= {
            f'dataset-{dataset_id}-image-{image_id}',
            f'dataset-{dataset_id}-extra-ref',
            f'dataset-{dataset_id}',
        }
        assert all(os.path.exists(destination)
                   for _src, _context, destination in calls)


def test_training_artifact_purge_moves_matches_to_trash(app, tmp_path, monkeypatch):
    from app import config as cfg
    from app.services import lora_training as training

    with app.app_context():
        aitoolkit = tmp_path / 'aitoolkit'
        output = aitoolkit / 'output'
        datasets = aitoolkit / 'datasets'
        jobs = aitoolkit / 'config' / 'generated'
        for directory in (output, datasets, jobs):
            directory.mkdir(parents=True)
        cfg.save_config({'aitoolkit': {'dir': str(aitoolkit)}})

        run_output = output / 'ulocal_TrashMe'
        run_dataset = datasets / 'ulocal_TrashMe_Krea-2-Raw'
        run_output.mkdir()
        run_dataset.mkdir()
        job = jobs / 'ulocal_TrashMe.json'
        job.write_text('{}', encoding='utf-8')
        sibling = jobs / 'ulocal_TrashMe2.json'
        sibling.write_text('{}', encoding='utf-8')

        calls = _trash_spy(monkeypatch)
        removed = training.purge_training_artifacts('local', 'TrashMe')

        assert set(removed) == {str(run_output), str(run_dataset), str(job)}
        assert not run_output.exists() and not run_dataset.exists() and not job.exists()
        assert sibling.exists()
        assert len(calls) == 3
        assert all(context == 'training-TrashMe'
                   for _src, context, _destination in calls)


# --- Windows sharing-violation robustness on delete (WinError 32) -------------
#
# A dataset freshly cleaned by the Klein engine can have one of its images still
# held open by an antivirus scan (Bitdefender ATD) the instant the user hits the
# trash button. On Windows that open handle blocks the folder move, and the delete
# used to bubble a bare PermissionError up to a 500 — while shutil.move's
# copytree+rmtree fallback left a full COPY of the "undeleted" dataset in Trash.

def _lock_file(path):
    """Open `path` the way Windows locks a file (no FILE_SHARE_DELETE), so a move
    of it — or of its parent folder — raises WinError 32/5, exactly like an
    antivirus mid-scan. A real handle: no mock, faithful on Windows."""
    return open(path, 'rb')


def test_send_to_trash_locked_file_aborts_without_partial_copy(app, monkeypatch):
    from app.services import trash

    with app.app_context():
        monkeypatch.setattr(trash, '_LOCK_RETRY_DELAY', 0.01)  # keep the test fast
        folder = trash.trash_root().parent / 'datasets' / 'locked'
        folder.mkdir(parents=True)
        locked = folder / 'shot.png'
        locked.write_bytes(b'pixels')
        before = set(trash.trash_root().iterdir())
        handle = _lock_file(locked)
        try:
            with pytest.raises(trash.TrashLockError):
                trash.send_to_trash(folder, context='dataset-locked')
        finally:
            handle.close()
        # Clean abort: source untouched, and NOTHING copied into Trash (no stray
        # staging dir left behind — the pre-fix bug left a full duplicate).
        assert locked.exists()
        assert set(trash.trash_root().iterdir()) == before
        # TrashLockError stays an OSError so blanket `except OSError` still catches.
        assert issubclass(trash.TrashLockError, OSError)


def test_send_to_trash_retries_over_a_transient_lock(app, monkeypatch):
    from app.services import trash

    with app.app_context():
        monkeypatch.setattr(trash, '_LOCK_RETRY_DELAY', 0.01)
        folder = trash.trash_root().parent / 'datasets' / 'transient'
        folder.mkdir(parents=True)
        locked = folder / 'shot.png'
        locked.write_bytes(b'pixels')
        handle = _lock_file(locked)
        # Release the handle during the first backoff: the next rename then wins.
        released = {'done': False}

        def release(_delay):
            if not released['done']:
                handle.close()
                released['done'] = True

        monkeypatch.setattr(trash.time, 'sleep', release)
        dest = trash.send_to_trash(folder, context='dataset-transient')
        assert os.path.exists(dest)
        assert not folder.exists()


def test_delete_dataset_locked_file_reports_instead_of_500(app, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.services import trash
    from app.extensions import db
    from app.models import FaceDatasetImage

    with app.app_context():
        monkeypatch.setattr(trash, '_LOCK_RETRY_DELAY', 0.01)
        ds = svc.create_dataset('local', 'Demo', 'demo')
        folder = svc._dataset_dir(ds.id)
        # Post-clean state: the cleaned image plus its .orig sibling on disk.
        shot = os.path.join(folder, 'shot.png')
        with open(shot, 'wb') as fh:
            fh.write(b'cleaned')
        with open(os.path.join(folder, 'shot.png.orig'), 'wb') as fh:
            fh.write(b'original')
        db.session.add(FaceDatasetImage(dataset_id=ds.id, filename='shot.png',
                                        status='keep', source='import'))
        db.session.commit()
        dataset_id = ds.id

        handle = _lock_file(shot)
        try:
            with pytest.raises(RuntimeError, match='still open in another program'):
                svc.delete_dataset('local', dataset_id)
        finally:
            handle.close()
        # Fully intact: nothing half-deleted on disk, DB row still there.
        assert os.path.isdir(folder)
        assert svc.get_dataset('local', dataset_id) is not None
        assert not any(trash.trash_root().iterdir())
        # Handle released -> a retry now trashes the whole folder (.orig included).
        assert svc.delete_dataset('local', dataset_id) is True
        assert not os.path.isdir(folder)
        assert svc.get_dataset('local', dataset_id) is None


# --- Refuse deleting a dataset with a training run mid-flight (409, not orphan) -
#
# Deleting a dataset while a run is training used to silently succeed: the folder
# went to Trash, the FaceDataset row vanished, and the CloudTrainingRun row was
# left with a dangling dataset_id while its paid vast pod kept training against
# images we just moved out from under it. delete_dataset now refuses with a
# RuntimeError the route maps to a 409, so nothing is touched until the run stops.

@pytest.mark.parametrize('active_state',
                         ['preparing', 'provisioning', 'uploading', 'training',
                          'downloading', 'terminating'])
def test_delete_dataset_refused_while_cloud_run_active(app, monkeypatch, active_state):
    from app.extensions import db
    from app.models import CloudTrainingRun
    from app.services import face_dataset_service as svc

    with app.app_context():
        calls = _trash_spy(monkeypatch)
        ds = svc.create_dataset('local', 'Demo', 'demotrig')
        folder = svc._dataset_dir(ds.id)
        with open(os.path.join(folder, 'shot.webp'), 'wb') as fh:
            fh.write(b'pixels')
        run = CloudTrainingRun(dataset_id=ds.id, status=active_state,
                               run_name='r', job_name='j')
        db.session.add(run)
        db.session.commit()
        dataset_id, run_id = ds.id, run.id

        with pytest.raises(RuntimeError, match='training run is active'):
            svc.delete_dataset('local', dataset_id)

        # Nothing touched: folder, dataset row and the run all intact, no trashing.
        assert os.path.isdir(folder)
        assert svc.get_dataset('local', dataset_id) is not None
        assert db.session.get(CloudTrainingRun, run_id) is not None
        assert calls == []


def test_delete_dataset_allowed_once_cloud_run_terminal(app, monkeypatch):
    """A finished run doesn't block delete, and its provenance rows survive with
    an orphaned dataset_id — run history stays after the dataset is gone."""
    from app.extensions import db
    from app.models import CloudTrainingRun, TrainingRunRecord
    from app.services import cloud_training as ct
    from app.services import face_dataset_service as svc

    with app.app_context():
        _trash_spy(monkeypatch)
        ds = svc.create_dataset('local', 'Demo', 'demotrig')
        folder = svc._dataset_dir(ds.id)
        with open(os.path.join(folder, 'shot.webp'), 'wb') as fh:
            fh.write(b'pixels')
        dataset_id = ds.id
        # One row per terminal state + a provenance record -> none should block.
        for st in ('done', 'stopped', 'error', 'error_pod_kept'):
            db.session.add(CloudTrainingRun(dataset_id=dataset_id, status=st,
                                            run_name=st, job_name=st))
        db.session.add(TrainingRunRecord(dataset_id=dataset_id, family='zimage',
                                         source='cloud', fingerprint='abc', version=1))
        db.session.commit()

        assert ct.active_runs_for(dataset_id) == []
        assert svc.delete_dataset('local', dataset_id) is True
        assert not os.path.isdir(folder)
        assert svc.get_dataset('local', dataset_id) is None
        # Provenance preserved (orphaned dataset_id) — the existing no-FK pattern.
        assert CloudTrainingRun.query.filter_by(dataset_id=dataset_id).count() == 4
        assert TrainingRunRecord.query.filter_by(dataset_id=dataset_id).count() == 1


def test_delete_dataset_refused_while_local_run_active(app, monkeypatch):
    from app.extensions import db
    from app.job_queue import queue_manager
    from app.services import face_dataset_service as svc

    with app.app_context():
        calls = _trash_spy(monkeypatch)
        ds = svc.create_dataset('local', 'Demo', 'demotrig')
        other = svc.create_dataset('local', 'Other', 'othertrig')
        dataset_id, other_id = ds.id, other.id

        queue_manager._set_system_state('training_in_progress', True)
        queue_manager._set_system_state('training_dataset_id', dataset_id)

        with pytest.raises(RuntimeError, match='training run is active'):
            svc.delete_dataset('local', dataset_id)
        assert svc.get_dataset('local', dataset_id) is not None
        assert calls == []

        # A run on a DIFFERENT dataset never blocks this one.
        assert svc.delete_dataset('local', other_id) is True

        # Run finished -> the original dataset deletes cleanly.
        queue_manager._set_system_state('training_in_progress', False)
        assert svc.delete_dataset('local', dataset_id) is True
        assert svc.get_dataset('local', dataset_id) is None


def test_delete_dataset_route_returns_409_for_active_run(client, app):
    """The route funnels the guard's RuntimeError through _map_error -> 409 with
    the message in the body, which the library toast surfaces verbatim."""
    from app.extensions import db
    from app.models import CloudTrainingRun
    from app.services import face_dataset_service as svc

    with app.app_context():
        ds = svc.create_dataset('local', 'Demo', 'demotrig')
        db.session.add(CloudTrainingRun(dataset_id=ds.id, status='training',
                                        run_name='r', job_name='j'))
        db.session.commit()
        dataset_id = ds.id

    resp = client.post(f'/api/dataset/{dataset_id}/delete')
    assert resp.status_code == 409
    assert 'training run is active on this dataset' in resp.get_json()['error']


# --- Delete on a LEGACY schema whose child FK lacks ON DELETE CASCADE ----------
#
# db.create_all() builds the child tables WITH ON DELETE CASCADE, but a DB first
# created by an older schema keeps its no-cascade tables forever (create_all never
# ALTERs an existing table). With PRAGMA foreign_keys=ON, deleting a parent that
# still has children then raises IntegrityError -> a bare 500 in prod. The child
# models declare only a table-level ForeignKey (no relationship()), so the unit of
# work has no ordering dependency and used to emit `DELETE FROM face_dataset`
# FIRST — before the explicit child deletes reached the DB. delete_dataset now
# flushes the children before the parent, so the belt works on every DB vintage.

def _rebuild_children_without_cascade(db):
    """Recreate the child tables from their real generated DDL with ON DELETE
    CASCADE stripped — identical columns, just the legacy (no-cascade) FK, exactly
    like a DB first created before the cascade was declared."""
    import re
    for table in ('lora_test_image', 'face_dataset_image'):
        ddl = db.session.execute(text(
            'SELECT sql FROM sqlite_master WHERE name = :n'), {'n': table}
        ).scalar_one()
        legacy = re.sub(r'\s+ON DELETE CASCADE', '', ddl, flags=re.I)
        assert 'CASCADE' not in legacy.upper()
        db.session.execute(text(f'DROP TABLE {table}'))
        db.session.execute(text(legacy))
    db.session.commit()
    # Fixture sanity: the FK exists but no longer cascades on delete.
    for table in ('lora_test_image', 'face_dataset_image'):
        fk = db.session.execute(text(
            f'PRAGMA foreign_key_list({table})')).all()
        assert any(row[2] == 'face_dataset' and row[6].upper() != 'CASCADE'
                   for row in fk)


def test_delete_dataset_legacy_no_cascade_with_studio_rows(app, monkeypatch):
    """Dataset 9 case: residual face_dataset_image AND lora_test_image on a
    no-cascade schema. The delete used to 500 on `DELETE FROM face_dataset`."""
    from app.extensions import db
    from app.models import FaceDatasetImage, LoraTestImage
    from app.services import face_dataset_service as svc

    with app.app_context():
        _trash_spy(monkeypatch)
        _rebuild_children_without_cascade(db)
        ds = svc.create_dataset('local', 'Nine', 'ninetrig')
        folder = svc._dataset_dir(ds.id)
        with open(os.path.join(folder, 'shot.webp'), 'wb') as fh:
            fh.write(b'pixels')
        for i in range(8):
            db.session.add(FaceDatasetImage(dataset_id=ds.id, filename=f'k{i}.webp',
                                            status='keep', source='import'))
        for i in range(20):
            db.session.add(LoraTestImage(dataset_id=ds.id, checkpoint='z image\\L.safetensors',
                                         strength=1.0, status='done'))
        db.session.commit()
        dataset_id = ds.id

        assert svc.delete_dataset('local', dataset_id) is True
        assert svc.get_dataset('local', dataset_id) is None
        assert FaceDatasetImage.query.filter_by(dataset_id=dataset_id).count() == 0
        assert LoraTestImage.query.filter_by(dataset_id=dataset_id).count() == 0
        assert not os.path.isdir(folder)


def test_delete_dataset_legacy_no_cascade_residual_images_only(app, monkeypatch):
    """Dataset 26 case reproduced: 15 residual face_dataset_image, 0 studio rows,
    on a no-cascade schema. The images survive the parent delete only because the
    parent DELETE was emitted first; flushing children first removes them cleanly."""
    from app.extensions import db
    from app.models import FaceDatasetImage, LoraTestImage
    from app.services import face_dataset_service as svc

    with app.app_context():
        _trash_spy(monkeypatch)
        _rebuild_children_without_cascade(db)
        ds = svc.create_dataset('local', 'TwentySix', 'ds26trig')
        for i in range(15):
            db.session.add(FaceDatasetImage(dataset_id=ds.id, filename=f'r{i}.webp',
                                            status='keep', source='import'))
        db.session.commit()
        dataset_id = ds.id
        assert FaceDatasetImage.query.filter_by(dataset_id=dataset_id).count() == 15

        assert svc.delete_dataset('local', dataset_id) is True
        assert svc.get_dataset('local', dataset_id) is None
        assert FaceDatasetImage.query.filter_by(dataset_id=dataset_id).count() == 0
        assert LoraTestImage.query.filter_by(dataset_id=dataset_id).count() == 0


def test_delete_dataset_route_no_500_on_legacy_no_cascade(client, app, monkeypatch):
    """End to end through the HTTP route: a legacy-schema dataset with children
    returns 200, never the bare 500 the unmapped IntegrityError used to produce."""
    from app.extensions import db
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc

    with app.app_context():
        _rebuild_children_without_cascade(db)
        ds = svc.create_dataset('local', 'RouteLegacy', 'routelegacy')
        db.session.add(FaceDatasetImage(dataset_id=ds.id, filename='a.webp',
                                        status='keep', source='import'))
        db.session.commit()
        dataset_id = ds.id

    resp = client.post(f'/api/dataset/{dataset_id}/delete')
    assert resp.status_code == 200
    assert resp.get_json() == {'ok': True}


def test_map_error_maps_integrity_error_to_409(app):
    """An IntegrityError that slips past the belt maps to a clear 409, never a
    bare 500 (the unmapped path that produced 'Server error (500)' in prod)."""
    from sqlalchemy.exc import IntegrityError
    from app.routes._common import _map_error

    with app.app_context():
        err = IntegrityError('DELETE FROM face_dataset WHERE face_dataset.id = ?',
                             params=(26,), orig=Exception('FOREIGN KEY constraint failed'))
        body, status = _map_error(err)
        assert status == 409
        payload = body.get_json()
        assert 'error' in payload
        assert 'conflict' in payload['error'].lower()
