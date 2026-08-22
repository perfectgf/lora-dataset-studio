"""Server-side ✨ Klein upscale & improve BATCH.

The batch used to be a loop in the browser, one request per image, which produced
two bugs with one root cause (the batch only existed in the tab):

* a selection bigger than MAX_FANOUT was mostly REFUSED — that cap is a
  CONCURRENCY limit, and a client loop that keeps pushing walks straight into it
  (250 selected -> 60 queued, 190 counted as failures);
* ⏹ Stop generation was powerless — it cancelled the rows in flight and the tab
  immediately queued the next wave.

These tests cover the fixed contract: the job drains the WHOLE selection in waves
under the cap, and Stop really ends it.
"""
import io
import os

from PIL import Image


def _png():
    buf = io.BytesIO()
    Image.new('RGB', (96, 64), (25, 50, 75)).save(buf, 'PNG')
    return buf.getvalue()


def _dataset_with_sources(svc, image_cls, user_id, count):
    """A dataset holding ``count`` kept, on-disk images eligible for improvement."""
    ds = svc.create_dataset(user_id, 'Bulk improve', 'improve')
    os.makedirs(svc._dataset_dir(ds.id), exist_ok=True)
    ids = []
    for i in range(count):
        filename = f'source-{i}.png'
        with open(os.path.join(svc._dataset_dir(ds.id), filename), 'wb') as fh:
            fh.write(_png())
        img = image_cls(dataset_id=ds.id, filename=filename, source='import',
                        status='keep', framing='body',
                        variation_label=f'Imported {i}')
        svc.db.session.add(img)
        svc.db.session.commit()
        ids.append(img.id)
    return ds, ids


def _stub_klein(monkeypatch, keh, jobs):
    monkeypatch.setattr(keh, 'klein_missing_assets', lambda: [])
    monkeypatch.setattr(keh, 'klein_missing_nodes', lambda: [])
    monkeypatch.setattr(keh, 'enqueue_klein_edit',
                        lambda **kwargs: (jobs.append(kwargs) or f'job-{len(jobs)}'))


def test_batch_larger_than_max_fanout_eventually_processes_everything(app, monkeypatch):
    """The wave mechanism: when the cap is reached the worker WAITS for a slot
    instead of firing a request doomed to be refused, and the whole selection is
    queued. Before the fix everything past MAX_FANOUT was counted as a failure."""
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import dataset_activity as da
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper as keh

    jobs = []
    _stub_klein(monkeypatch, keh, jobs)
    monkeypatch.setattr(svc, 'MAX_FANOUT', 3)   # a cap we can walk into in a test

    with app.app_context():
        da.reset()
        ds, source_ids = _dataset_with_sources(svc, FaceDatasetImage, LOCAL_USER, 10)
        token = da.begin(ds.id, 'improve', total=len(source_ids))
        peaks, waits = [], []

        def _free_one_slot(_seconds):
            """Stand-in for the wait: ComfyUI delivering one file frees one slot."""
            waits.append(_seconds)
            oldest = (FaceDatasetImage.query
                      .filter_by(dataset_id=ds.id, status='pending')
                      .filter(FaceDatasetImage.filename.is_(None))
                      .order_by(FaceDatasetImage.id.asc()).first())
            assert oldest is not None, 'waiting on an empty queue would never end'
            oldest.filename = f'improved-{oldest.id}.png'
            svc.db.session.commit()

        real_improve = svc.improve_existing_image

        def _tracked(user_id, image_id, engine=None):
            result = real_improve(user_id, image_id, engine=engine)
            peaks.append(svc._improve_in_flight(ds.id))
            return result

        monkeypatch.setattr(svc, 'improve_existing_image', _tracked)
        summary = svc._drain_improve_queue(LOCAL_USER, ds.id, source_ids, token,
                                           sleep=_free_one_slot)
        da.end(token)

    assert summary == {'total': 10, 'queued': 10, 'failed': 0,
                       'stopped': False, 'stalled': False, 'remaining': 0}
    assert len(jobs) == 10                 # every image really reached the queue
    assert max(peaks) <= 3                 # …without ever exceeding the cap
    assert len(waits) == 7                 # the 7 images past the cap really WAITED
    assert waits == [svc.IMPROVE_SLOT_POLL_SECONDS] * 7


def test_stop_generation_really_ends_the_batch(app, monkeypatch):
    """⏹ Stop generation (cancel_pending) arms the cooperative flag the worker polls,
    so the batch stops at the next image instead of re-queuing another wave."""
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import dataset_activity as da
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper as keh

    jobs = []
    _stub_klein(monkeypatch, keh, jobs)

    with app.app_context():
        da.reset()
        ds, source_ids = _dataset_with_sources(svc, FaceDatasetImage, LOCAL_USER, 8)
        token = da.begin(ds.id, 'improve', total=len(source_ids))
        real_improve = svc.improve_existing_image

        def _stop_after_three(user_id, image_id, engine=None):
            result = real_improve(user_id, image_id, engine=engine)
            if len(jobs) == 3:
                svc.cancel_pending(LOCAL_USER, ds.id)   # the ⏹ Stop button path
            return result

        monkeypatch.setattr(svc, 'improve_existing_image', _stop_after_three)
        summary = svc._drain_improve_queue(LOCAL_USER, ds.id, source_ids, token)
        assert da.cancel_requested(ds.id, da.IMPROVE_KINDS) is True
        da.end(token)
        da.clear_cancel(ds.id, da.IMPROVE_KINDS)

    assert summary['stopped'] is True
    assert summary['queued'] == 3
    assert summary['remaining'] == 5
    assert len(jobs) == 3      # nothing was queued after the Stop


def test_stop_generation_does_not_stop_a_captioning_batch(app):
    """The two Stop buttons have separate arming scopes: stopping generations must
    not silently end a captioning pass (and vice versa)."""
    from app.services import dataset_activity as da

    da.reset()
    caption_token = da.begin(41, 'caption', total=5)
    improve_token = da.begin(41, 'improve', total=5)
    assert da.request_cancel(41, da.IMPROVE_KINDS) is True
    assert da.cancel_requested(41, da.IMPROVE_KINDS) is True
    assert da.cancel_requested(41) is False           # caption scope untouched
    assert da.request_cancel(41) is True              # now stop the caption pass
    assert da.cancel_requested(41) is True
    da.end(caption_token)
    da.end(improve_token)
    da.reset()


def test_activity_prefers_the_batch_handle_over_the_in_flight_count(app):
    """A worker-owned entry beats the count-derived one: the improve batch knows the
    honest total (the whole selection), the synced entry only sees what is in flight."""
    from app.services import dataset_activity as da

    da.reset()
    token = da.begin(42, 'improve', total=250, detail='Queuing improvements… 0/250')
    da.bump(token, 12)
    da.sync_pending(42, 'generate', 60, engine='klein')   # started LATER
    activity = da.get(42)
    assert activity['kind'] == 'improve'
    assert (activity['done'], activity['total']) == (12, 250)
    da.end(token)
    da.reset()


def test_eligible_ids_mirror_the_client_partition(app, monkeypatch):
    """The job announces the number it will really work on — ineligible rows are
    dropped up front rather than refused one by one inside the loop."""
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc

    with app.app_context():
        ds, source_ids = _dataset_with_sources(svc, FaceDatasetImage, LOCAL_USER, 3)
        fileless = FaceDatasetImage(dataset_id=ds.id, source='generated',
                                    status='pending')
        already = FaceDatasetImage(dataset_id=ds.id, source='generated', status='keep',
                                   filename='x.png',
                                   derivation_kind=svc.KLEIN_IMAGE_IMPROVE)
        svc.db.session.add_all([fileless, already])
        svc.db.session.commit()
        # The first source already has an improvement awaiting review.
        pending_child = FaceDatasetImage(dataset_id=ds.id, source='generated',
                                         status='pending',
                                         parent_image_id=source_ids[0],
                                         derivation_kind=svc.KLEIN_IMAGE_IMPROVE)
        svc.db.session.add(pending_child)
        svc.db.session.commit()

        selection = [*source_ids, fileless.id, already.id, 999999, source_ids[1]]
        assert svc.bulk_improve_eligible_ids(LOCAL_USER, ds.id, selection) == source_ids[1:]


def test_route_starts_the_job_and_refuses_a_second_one(app, client, monkeypatch):
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import dataset_activity as da
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper as keh

    jobs = []
    _stub_klein(monkeypatch, keh, jobs)

    with app.app_context():
        da.reset()
        ds, source_ids = _dataset_with_sources(svc, FaceDatasetImage, LOCAL_USER, 4)
        dataset_id = ds.id

    resp = client.post(f'/api/dataset/{dataset_id}/improve/batch',
                       json={'image_ids': [*source_ids, 424242]})
    assert resp.status_code == 200
    assert resp.get_json() == {'ok': True, 'queued': 4, 'skipped': 1,
                               'engine': 'klein'}
    assert len(jobs) == 4

    # A live batch refuses a second one (409) — two workers racing the same cap
    # would defeat the point of the wave loop. Under TESTING the job ran inline, so
    # simulate the live entry explicitly.
    with app.app_context():
        token = da.begin(dataset_id, 'improve', total=4)
    conflict = client.post(f'/api/dataset/{dataset_id}/improve/batch',
                           json={'image_ids': source_ids})
    assert conflict.status_code == 409
    with app.app_context():
        da.end(token)
        da.reset()

    bad = client.post(f'/api/dataset/{dataset_id}/improve/batch', json={})
    assert bad.status_code == 400


def test_batch_route_recovery_barrier_has_no_service_side_effect(
        client, monkeypatch):
    from app.job_queue import COMFYUI_STALLED_BARRIER_KEY, queue_manager
    from app.services import face_dataset_service as svc

    with client.application.app_context():
        queue_manager._set_system_state(
            COMFYUI_STALLED_BARRIER_KEY, {'job_id': 'unresolved'})
    monkeypatch.setattr(
        svc, 'start_bulk_improve',
        lambda *_args: (_ for _ in ()).throw(AssertionError('service must not run')))

    response = client.post(
        '/api/dataset/1/improve/batch', json={'image_ids': [1]})
    assert response.status_code == 409
    assert response.get_json()['code'] == 'comfyui_recovery_required'


def test_the_batch_leaves_the_fanout_budget_to_the_user(app, monkeypatch):
    """GitHub #44, the half a grey button would not have fixed.

    The drain used to queue until the whole per-dataset budget was spent, so the
    ⚡ Generate the user clicked next was refused ("too many generations in
    flight"). It now keeps only IMPROVE_QUEUE_DEPTH of its own work in flight,
    which is all a one-job-at-a-time ComfyUI can use anyway, and the rest of
    MAX_FANOUT stays available for whatever the person launches."""
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import dataset_activity as da
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper as keh

    jobs = []
    _stub_klein(monkeypatch, keh, jobs)
    monkeypatch.setattr(svc, 'MAX_FANOUT', 20)
    monkeypatch.setattr(svc, 'IMPROVE_QUEUE_DEPTH', 3)

    with app.app_context():
        da.reset()
        ds, source_ids = _dataset_with_sources(svc, FaceDatasetImage, LOCAL_USER, 9)
        token = da.begin(ds.id, 'improve', total=len(source_ids))

        def _free_one_slot(_seconds):
            oldest = (FaceDatasetImage.query
                      .filter_by(dataset_id=ds.id,
                                 derivation_kind=svc.KLEIN_IMAGE_IMPROVE,
                                 status='pending')
                      .filter(FaceDatasetImage.filename.is_(None))
                      .order_by(FaceDatasetImage.id.asc()).first())
            assert oldest is not None, 'waiting on an empty queue would never end'
            oldest.filename = f'improved-{oldest.id}.png'
            svc.db.session.commit()

        headroom = []
        real_improve = svc.improve_existing_image

        def _tracked(user_id, image_id, engine=None):
            result = real_improve(user_id, image_id, engine=engine)
            # What a ⚡ Generate launched at this exact moment could still ask for.
            headroom.append(svc.MAX_FANOUT - svc._improve_in_flight(ds.id))
            return result

        monkeypatch.setattr(svc, 'improve_existing_image', _tracked)
        summary = svc._drain_improve_queue(LOCAL_USER, ds.id, source_ids, token,
                                           sleep=_free_one_slot)
        da.end(token)

    assert summary['queued'] == 9 and summary['failed'] == 0
    assert len(jobs) == 9                     # the whole selection still gets queued
    # The batch never holds more than its depth, so the user keeps 17 of the 20.
    assert max(9 - h for h in headroom) <= 3
    assert min(headroom) == 17


def test_the_batch_does_not_wait_on_generations_that_are_not_its_own(app, monkeypatch):
    """A user's own ⚡ Generate batch used to park the drain: the wait was measured
    over EVERY unfinished generation on the dataset, not the improvements. With a
    cap of 4 and three interactive generations pending, the old condition queued
    one improvement and then waited on rows it does not own and cannot free."""
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import dataset_activity as da
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper as keh

    jobs = []
    _stub_klein(monkeypatch, keh, jobs)
    monkeypatch.setattr(svc, 'MAX_FANOUT', 8)
    monkeypatch.setattr(svc, 'IMPROVE_QUEUE_DEPTH', 4)

    with app.app_context():
        da.reset()
        ds, source_ids = _dataset_with_sources(svc, FaceDatasetImage, LOCAL_USER, 4)
        # Three interactive generations already in flight (no file yet), exactly
        # what ⚡ Generate leaves behind while ComfyUI works through them.
        for i in range(3):
            svc.db.session.add(FaceDatasetImage(
                dataset_id=ds.id, filename=None, source='generated',
                status='pending', variation_label=f'In flight {i}'))
        svc.db.session.commit()
        assert svc._improve_in_flight(ds.id) == 3
        assert svc._improve_batch_in_flight(ds.id) == 0

        token = da.begin(ds.id, 'improve', total=len(source_ids))
        waits = []
        summary = svc._drain_improve_queue(
            LOCAL_USER, ds.id, source_ids, token,
            sleep=lambda seconds: waits.append(seconds))
        da.end(token)

    # 3 foreign + 4 own = 7, under the cap of 8: nothing had to wait.
    assert summary['queued'] == 4 and summary['stalled'] is False
    assert waits == []


def test_a_batch_is_not_declared_stalled_while_the_queue_is_held_by_training(app, monkeypatch):
    """The regression the shallower wave depth opened.

    With the old depth (MAX_FANOUT, 60) an ordinary batch never entered the wait
    loop at all. At IMPROVE_QUEUE_DEPTH the loop is the NORMAL path — so a
    training run holding the GPU for longer than IMPROVE_SLOT_TIMEOUT_SECONDS
    made the drain declare itself stalled and drop every image it had not queued
    yet, silently. Time spent waiting on a queue that is provably held is not a
    stall; it is waiting."""
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import dataset_activity as da
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper as keh

    jobs = []
    _stub_klein(monkeypatch, keh, jobs)
    monkeypatch.setattr(svc, 'MAX_FANOUT', 20)
    monkeypatch.setattr(svc, 'IMPROVE_QUEUE_DEPTH', 2)

    with app.app_context():
        da.reset()
        ds, source_ids = _dataset_with_sources(svc, FaceDatasetImage, LOCAL_USER, 6)
        token = da.begin(ds.id, 'improve', total=len(source_ids))

        # The GPU is held by training for far longer than the stall timeout, then
        # released. Nothing frees a slot in the meantime — which is the point.
        held = {'polls': 0}
        monkeypatch.setattr(svc, '_queue_held_off_gpu_reason',
                            lambda: 'LoRA training' if held['polls'] < 12 else None)

        def _tick(_seconds):
            held['polls'] += 1
            if held['polls'] < 12:
                return                      # frozen: no slot frees
            oldest = (FaceDatasetImage.query
                      .filter_by(dataset_id=ds.id,
                                 derivation_kind=svc.KLEIN_IMAGE_IMPROVE,
                                 status='pending')
                      .filter(FaceDatasetImage.filename.is_(None))
                      .order_by(FaceDatasetImage.id.asc()).first())
            if oldest is not None:
                oldest.filename = f'improved-{oldest.id}.png'
                svc.db.session.commit()

        # Long enough that the OLD accounting would have timed out several times.
        monkeypatch.setattr(svc, 'IMPROVE_SLOT_TIMEOUT_SECONDS', 10.0)
        monkeypatch.setattr(svc, 'IMPROVE_SLOT_POLL_SECONDS', 2.0)
        summary = svc._drain_improve_queue(LOCAL_USER, ds.id, source_ids, token,
                                           sleep=_tick)
        da.end(token)

    assert summary['stalled'] is False, 'a held queue was counted as a stall'
    assert summary['queued'] == 6 and summary['remaining'] == 0
    assert len(jobs) == 6


def test_a_batch_still_gives_up_when_nothing_holds_the_queue_and_no_slot_frees(app, monkeypatch):
    """The other half: the stall timeout must still exist. A ComfyUI that died
    mid-batch frees no slot and reports no hold — that one is a real stall, and
    the drain has to stop rather than poll a count that never drops."""
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import dataset_activity as da
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper as keh

    jobs = []
    _stub_klein(monkeypatch, keh, jobs)
    monkeypatch.setattr(svc, 'MAX_FANOUT', 20)
    monkeypatch.setattr(svc, 'IMPROVE_QUEUE_DEPTH', 2)
    monkeypatch.setattr(svc, 'IMPROVE_SLOT_TIMEOUT_SECONDS', 6.0)
    monkeypatch.setattr(svc, 'IMPROVE_SLOT_POLL_SECONDS', 2.0)
    monkeypatch.setattr(svc, '_queue_held_off_gpu_reason', lambda: None)

    with app.app_context():
        da.reset()
        ds, source_ids = _dataset_with_sources(svc, FaceDatasetImage, LOCAL_USER, 6)
        token = da.begin(ds.id, 'improve', total=len(source_ids))
        summary = svc._drain_improve_queue(LOCAL_USER, ds.id, source_ids, token,
                                           sleep=lambda _s: None)
        da.end(token)

    assert summary['stalled'] is True
    assert summary['queued'] == 2 and summary['remaining'] == 4
