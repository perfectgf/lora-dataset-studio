"""Re-running the ✨ Upscale & improve pass on a tile that IS an improvement.

There was no way to do it: the tile hides the generic 🔄 on purpose (that route
restarts from the DATASET REFERENCE and the catalog prompt, so on an improved
image it would quietly produce an unrelated variation), and the improve settings
became editable — so the only way to see a new value take effect was to delete
the result and click ✨ again on the parent.

`reimprove_image` is that gesture: same parent, current settings, result replaced
in place. These tests pin the four things that make it correct — it starts from
the PARENT, it reads TODAY's config, the generic route stays closed, and a
deleted parent refuses instead of silently improving something else.
"""
import io
import os

import pytest
from PIL import Image


def _png(color=(25, 50, 75)):
    buf = io.BytesIO()
    Image.new('RGB', (96, 64), color).save(buf, 'PNG')
    return buf.getvalue()


def _stub_klein(monkeypatch, keh, queued):
    monkeypatch.setattr(keh, 'klein_missing_assets', lambda: [])
    monkeypatch.setattr(keh, 'klein_missing_nodes', lambda: [])
    monkeypatch.setattr(
        keh, 'enqueue_klein_edit',
        lambda **kwargs: (queued.append(kwargs) or f'improve-job-{len(queued)}'))


def _improved_pair(svc, image_cls, user_id, *, parent_filename='parent.png',
                   candidate_filename='improved.png'):
    """A parent image and the improvement derived from it, both on disk.

    Built the way a row written BEFORE this feature looks: parent_image_id +
    derivation_kind and nothing else — which is exactly what makes the re-run
    available to results improved by earlier versions.
    """
    ds = svc.create_dataset(user_id, 'Improve', 'improve')
    os.makedirs(svc._dataset_dir(ds.id), exist_ok=True)
    for filename, color in ((parent_filename, (25, 50, 75)),
                            (candidate_filename, (200, 180, 160))):
        if filename:
            with open(os.path.join(svc._dataset_dir(ds.id), filename), 'wb') as fh:
                fh.write(_png(color))
    parent = image_cls(dataset_id=ds.id, filename=parent_filename, source='import',
                       status='keep', framing='body', caption='full body, outdoor light',
                       variation_label='Imported low-resolution image')
    svc.db.session.add(parent)
    svc.db.session.commit()
    candidate = image_cls(dataset_id=ds.id, filename=candidate_filename,
                          source='generated', status='pending', framing='body',
                          caption='full body, outdoor light',
                          parent_image_id=parent.id,
                          derivation_kind=svc.KLEIN_IMAGE_IMPROVE,
                          variation_label='Klein upscale & improve · Imported low-resolution image',
                          variation_prompt='the instruction of the day it first ran',
                          job_id='improve-job-0')
    svc.db.session.add(candidate)
    svc.db.session.commit()
    return ds, parent, candidate


@pytest.mark.plugins('image_upscale')
def test_reimprove_reruns_the_pass_from_the_parent_with_todays_settings(app, monkeypatch):
    """The whole point: the pass restarts from the PARENT's pixels (never the
    dataset reference), with the improve knobs as they are NOW — a user who edits
    klein.improve_steps must be able to see the effect without deleting the tile."""
    from app import config as app_cfg
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper as keh

    queued, trashed = [], []
    _stub_klein(monkeypatch, keh, queued)
    monkeypatch.setattr(svc.trash, 'send_to_trash',
                        lambda path, context=None: trashed.append(path))
    monkeypatch.setattr(svc, '_sync_generate_activity', lambda dataset_id: None)

    with app.app_context():
        ds, parent, candidate = _improved_pair(svc, FaceDatasetImage, LOCAL_USER)
        candidate_id, parent_id = candidate.id, parent.id
        parent_path = svc._img_path(parent)
        old_candidate_path = svc._img_path(candidate)
        # The user tunes the pass AFTER the first result exists.
        app_cfg.save_config({'klein': {'improve_steps': 9,
                                       'improve_megapixels': 4.0}})

        result = svc.reimprove_image(LOCAL_USER, candidate_id)

        assert result['candidate_id'] == candidate_id, 'the tile is replaced in place'
        assert len(queued) == 1
        job = queued[0]
        assert job['source_path'] == parent_path, 'the pass must start from the parent'
        assert job['source_filename'] == parent.filename
        assert job['sampler_steps'] == 9, 'current setting, not the one used first time'
        assert job['output_megapixels'] == 4.0
        assert job['extra_metadata']['parent_image_id'] == parent_id
        assert job['extra_metadata']['derivation_kind'] == svc.KLEIN_IMAGE_IMPROVE

        svc.db.session.expire_all()
        row = svc.db.session.get(FaceDatasetImage, candidate_id)
        assert row.filename is None and row.status == 'pending'
        assert row.job_id == result['job_id'] != 'improve-job-0'
        assert row.parent_image_id == parent_id
        assert row.derivation_kind == svc.KLEIN_IMAGE_IMPROVE
        assert row.caption == 'full body, outdoor light', 'a typed caption survives'
        # The superseded result leaves cleanly — row, then file.
        assert trashed == [old_candidate_path]

        # The parent is untouched: same file, same row.
        parent = svc.db.session.get(FaceDatasetImage, parent_id)
        assert parent.filename and os.path.isfile(svc._img_path(parent))
        assert parent.status == 'keep'


@pytest.mark.plugins('image_upscale')
def test_reimprove_kept_candidate_restores_parent_as_fallback_after_late_failure(
        app, monkeypatch):
    """A re-run replaces a selected result with an in-flight row.

    Its parent must cover that gap, including when the new ComfyUI job fails,
    instead of leaving no kept image for the source shot.
    """
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper as keh

    queued = []
    _stub_klein(monkeypatch, keh, queued)
    monkeypatch.setattr(svc.trash, 'send_to_trash', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(svc, '_sync_generate_activity', lambda _dataset_id: None)

    with app.app_context():
        ds, parent, candidate = _improved_pair(svc, FaceDatasetImage, LOCAL_USER)
        parent.status = 'pending'
        candidate.status = 'keep'
        svc.db.session.commit()

        result = svc.reimprove_image(LOCAL_USER, candidate.id)

        svc.db.session.expire_all()
        assert svc.db.session.get(FaceDatasetImage, candidate.id).status == 'pending'
        assert svc.db.session.get(FaceDatasetImage, candidate.id).filename is None
        assert svc.db.session.get(FaceDatasetImage, parent.id).status == 'keep'

        svc.link_completed_dataset_image(result['job_id'], 'rerun-failed.png', failed=True)
        svc.db.session.expire_all()
        assert svc.db.session.get(FaceDatasetImage, candidate.id).status == 'failed'
        assert svc.db.session.get(FaceDatasetImage, parent.id).status == 'keep'


@pytest.mark.plugins('image_upscale')
def test_reimprove_trash_failure_restores_candidate_and_parent_states(app, monkeypatch):
    from app import job_queue
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper as keh

    queued = []
    _stub_klein(monkeypatch, keh, queued)
    monkeypatch.setattr(
        svc.trash, 'send_to_trash',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError('Trash unavailable')))
    monkeypatch.setattr(job_queue.queue_manager, 'cancel_job', lambda *_args, **_kwargs: True)

    with app.app_context():
        ds, parent, candidate = _improved_pair(svc, FaceDatasetImage, LOCAL_USER)
        parent.status = 'pending'
        candidate.status = 'keep'
        svc.db.session.commit()
        candidate_id, parent_id = candidate.id, parent.id

        with pytest.raises(OSError, match='Trash unavailable'):
            svc.reimprove_image(LOCAL_USER, candidate_id)

        svc.db.session.expire_all()
        restored_candidate = svc.db.session.get(FaceDatasetImage, candidate_id)
        assert restored_candidate.status == 'keep'
        assert restored_candidate.filename == 'improved.png'
        assert svc.db.session.get(FaceDatasetImage, parent_id).status == 'pending'


@pytest.mark.plugins('image_upscale')
def test_reimprove_never_overwrites_a_candidate_changed_during_enqueue(app, monkeypatch):
    """A status click racing the queue hand-off wins over the re-run request."""
    from sqlalchemy import update

    from app import job_queue
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper as keh

    cancelled, trashed = [], []
    monkeypatch.setattr(keh, 'klein_missing_assets', lambda: [])
    monkeypatch.setattr(keh, 'klein_missing_nodes', lambda: [])
    monkeypatch.setattr(job_queue.queue_manager, 'cancel_job',
                        lambda *args, **_kwargs: cancelled.append(args))
    monkeypatch.setattr(svc.trash, 'send_to_trash',
                        lambda path, context=None: trashed.append(path))

    with app.app_context():
        ds, parent, candidate = _improved_pair(svc, FaceDatasetImage, LOCAL_USER)
        parent.status = 'pending'
        candidate.status = 'keep'
        svc.db.session.commit()
        candidate_id, parent_id = candidate.id, parent.id

        def enqueue(**_kwargs):
            # Simulate a second request committing a Reject while this request
            # is still waiting for ComfyUI to accept the new job.
            svc.db.session.execute(
                update(FaceDatasetImage)
                .where(FaceDatasetImage.id == candidate_id)
                .values(status='reject')
                .execution_options(synchronize_session=False))
            svc.db.session.commit()
            return 'race-job'

        monkeypatch.setattr(keh, 'enqueue_klein_edit', enqueue)
        with pytest.raises(RuntimeError, match='changed while it was being re-queued'):
            svc.reimprove_image(LOCAL_USER, candidate_id)

        svc.db.session.expire_all()
        assert svc.db.session.get(FaceDatasetImage, candidate_id).status == 'reject'
        assert svc.db.session.get(FaceDatasetImage, parent_id).status == 'pending'
        assert trashed == []
        assert cancelled and cancelled[0][0] == 'race-job'


@pytest.mark.plugins('image_upscale')
@pytest.mark.parametrize('old_caption', [None, 'caption before re-run'])
def test_reimprove_preserves_a_caption_edited_during_enqueue(app, monkeypatch, old_caption):
    from sqlalchemy import update

    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper as keh

    monkeypatch.setattr(keh, 'klein_missing_assets', lambda: [])
    monkeypatch.setattr(keh, 'klein_missing_nodes', lambda: [])
    monkeypatch.setattr(svc.trash, 'send_to_trash', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(svc, '_sync_generate_activity', lambda _dataset_id: None)

    with app.app_context():
        ds, parent, candidate = _improved_pair(svc, FaceDatasetImage, LOCAL_USER)
        candidate.caption = old_caption
        svc.db.session.commit()
        candidate_id = candidate.id

        def enqueue(**_kwargs):
            svc.db.session.execute(
                update(FaceDatasetImage)
                .where(FaceDatasetImage.id == candidate_id)
                .values(caption='caption edited while enqueueing')
                .execution_options(synchronize_session=False))
            svc.db.session.commit()
            return 'caption-race-job'

        monkeypatch.setattr(keh, 'enqueue_klein_edit', enqueue)
        result = svc.reimprove_image(LOCAL_USER, candidate_id)

        svc.db.session.expire_all()
        row = svc.db.session.get(FaceDatasetImage, candidate_id)
        assert row.status == 'pending'
        assert row.job_id == result['job_id'] == 'caption-race-job'
        assert row.caption == 'caption edited while enqueueing'


@pytest.mark.plugins('image_upscale')
def test_reimprove_trash_rollback_preserves_a_concurrent_candidate_decision(
        app, monkeypatch):
    from sqlalchemy import update

    from app import job_queue
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper as keh

    queued = []
    _stub_klein(monkeypatch, keh, queued)
    monkeypatch.setattr(job_queue.queue_manager, 'cancel_job', lambda *_args, **_kwargs: True)

    with app.app_context():
        ds, parent, candidate = _improved_pair(svc, FaceDatasetImage, LOCAL_USER)
        parent.status = 'pending'
        candidate.status = 'keep'
        svc.db.session.commit()
        candidate_id, parent_id = candidate.id, parent.id

        def trash_then_user_rejects(*_args, **_kwargs):
            svc.db.session.execute(
                update(FaceDatasetImage)
                .where(FaceDatasetImage.id == candidate_id)
                .values(status='reject')
                .execution_options(synchronize_session=False))
            svc.db.session.commit()
            raise OSError('Trash unavailable')

        monkeypatch.setattr(svc.trash, 'send_to_trash', trash_then_user_rejects)
        with pytest.raises(OSError, match='Trash unavailable'):
            svc.reimprove_image(LOCAL_USER, candidate_id)

        svc.db.session.expire_all()
        assert svc.db.session.get(FaceDatasetImage, candidate_id).status == 'reject'
        assert svc.db.session.get(FaceDatasetImage, parent_id).status == 'keep'


@pytest.mark.plugins('image_upscale')
def test_reimprove_trash_rollback_preserves_a_caption_edited_during_trash(
        app, monkeypatch):
    from sqlalchemy import update

    from app import job_queue
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper as keh

    queued = []
    _stub_klein(monkeypatch, keh, queued)
    monkeypatch.setattr(job_queue.queue_manager, 'cancel_job', lambda *_args, **_kwargs: True)

    with app.app_context():
        ds, parent, candidate = _improved_pair(svc, FaceDatasetImage, LOCAL_USER)
        parent.status = 'pending'
        candidate.status = 'keep'
        svc.db.session.commit()
        candidate_id, parent_id = candidate.id, parent.id

        def trash_then_caption_edit(*_args, **_kwargs):
            svc.db.session.execute(
                update(FaceDatasetImage)
                .where(FaceDatasetImage.id == candidate_id)
                .values(caption='caption edited while moving old result')
                .execution_options(synchronize_session=False))
            svc.db.session.commit()
            raise OSError('Trash unavailable')

        monkeypatch.setattr(svc.trash, 'send_to_trash', trash_then_caption_edit)
        with pytest.raises(OSError, match='Trash unavailable'):
            svc.reimprove_image(LOCAL_USER, candidate_id)

        svc.db.session.expire_all()
        row = svc.db.session.get(FaceDatasetImage, candidate_id)
        assert row.status == 'keep'
        assert row.filename == 'improved.png'
        assert row.caption == 'caption edited while moving old result'
        assert svc.db.session.get(FaceDatasetImage, parent_id).status == 'pending'


@pytest.mark.plugins('image_upscale')
def test_reimprove_trash_rollback_restores_an_unchanged_blank_caption(app, monkeypatch):
    from app import job_queue
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper as keh

    queued = []
    _stub_klein(monkeypatch, keh, queued)
    monkeypatch.setattr(job_queue.queue_manager, 'cancel_job', lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        svc.trash, 'send_to_trash',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError('Trash unavailable')))

    with app.app_context():
        ds, parent, candidate = _improved_pair(svc, FaceDatasetImage, LOCAL_USER)
        parent.status = 'pending'
        candidate.status = 'keep'
        candidate.caption = None
        svc.db.session.commit()
        candidate_id, parent_id = candidate.id, parent.id

        with pytest.raises(OSError, match='Trash unavailable'):
            svc.reimprove_image(LOCAL_USER, candidate_id)

        svc.db.session.expire_all()
        row = svc.db.session.get(FaceDatasetImage, candidate_id)
        assert row.status == 'keep'
        assert row.filename == 'improved.png'
        assert row.caption is None
        assert svc.db.session.get(FaceDatasetImage, parent_id).status == 'pending'


def test_generic_regenerate_still_refuses_an_improvement(app, monkeypatch):
    """The original guard must survive: the generic route would restart from the
    dataset reference and make an unrelated image. It stays closed, and now there
    is a correct action to point at instead."""
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper as keh

    queued = []
    _stub_klein(monkeypatch, keh, queued)

    with app.app_context():
        ds, parent, candidate = _improved_pair(svc, FaceDatasetImage, LOCAL_USER)
        with pytest.raises(ValueError, match='cannot be regenerated'):
            svc.regenerate_image(LOCAL_USER, candidate.id)
        # A camera view was made from one picture too.
        candidate.derivation_kind = svc.CAMERA_ANGLE
        svc.db.session.commit()
        with pytest.raises(ValueError, match='cannot be regenerated'):
            svc.regenerate_image(LOCAL_USER, candidate.id)
        assert queued == [], 'nothing may be enqueued through the generic route'


def test_reimprove_refuses_when_the_parent_is_gone(app, monkeypatch):
    """The link carries no ForeignKey (legacy databases), so a deleted source
    leaves a dangling id. Refuse and say so — improving "something" would be a
    silent lie, and the tile keeps its current image."""
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper as keh

    queued, trashed = [], []
    _stub_klein(monkeypatch, keh, queued)
    monkeypatch.setattr(svc.trash, 'send_to_trash',
                        lambda path, context=None: trashed.append(path))

    with app.app_context():
        ds, parent, candidate = _improved_pair(svc, FaceDatasetImage, LOCAL_USER)
        candidate_id = candidate.id
        candidate_path = svc._img_path(candidate)
        svc.db.session.delete(parent)
        svc.db.session.commit()

        with pytest.raises(ValueError, match='was deleted'):
            svc.reimprove_image(LOCAL_USER, candidate_id)

        assert queued == [] and trashed == []
        svc.db.session.expire_all()
        row = svc.db.session.get(FaceDatasetImage, candidate_id)
        assert row.filename and os.path.isfile(candidate_path), 'the tile is untouched'
        assert row.status == 'pending' and row.job_id == 'improve-job-0'


def test_reimprove_refuses_while_the_pass_is_still_running(app, monkeypatch):
    """A row with no file yet is mid-generation: a second job would fight the first
    for the same tile."""
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper as keh

    queued = []
    _stub_klein(monkeypatch, keh, queued)

    with app.app_context():
        ds, parent, candidate = _improved_pair(svc, FaceDatasetImage, LOCAL_USER,
                                               candidate_filename=None)
        candidate.filename = None
        svc.db.session.commit()
        with pytest.raises(RuntimeError, match='still generating'):
            svc.reimprove_image(LOCAL_USER, candidate.id)
        assert queued == []


def test_reimprove_refuses_a_row_that_is_not_an_improvement(app, monkeypatch):
    """Only improvements have a parent pass to re-run. A plain imported or
    generated row keeps the generic regenerate."""
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper as keh

    queued = []
    _stub_klein(monkeypatch, keh, queued)

    with app.app_context():
        ds, parent, candidate = _improved_pair(svc, FaceDatasetImage, LOCAL_USER)
        with pytest.raises(ValueError, match='only an upscale & improve result'):
            svc.reimprove_image(LOCAL_USER, parent.id)
        assert queued == []


def test_reimprove_route_answers_404_for_an_unknown_image(app, client):
    r = client.post('/api/dataset/image/999999/reimprove')
    assert r.status_code == 404


def test_reimprove_route_recovery_barrier_has_no_service_side_effect(
        client, monkeypatch):
    from app.job_queue import COMFYUI_STALLED_BARRIER_KEY, queue_manager
    from app.services import face_dataset_service as svc

    with client.application.app_context():
        queue_manager._set_system_state(
            COMFYUI_STALLED_BARRIER_KEY, {'job_id': 'unresolved'})
    monkeypatch.setattr(
        svc, 'reimprove_image',
        lambda *_args: (_ for _ in ()).throw(AssertionError('service must not run')))

    response = client.post('/api/dataset/image/7/reimprove')
    assert response.status_code == 409
    assert response.get_json()['code'] == 'comfyui_recovery_required'
