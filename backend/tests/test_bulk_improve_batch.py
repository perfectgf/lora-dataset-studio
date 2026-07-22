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

        def _tracked(user_id, image_id):
            result = real_improve(user_id, image_id)
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

        def _stop_after_three(user_id, image_id):
            result = real_improve(user_id, image_id)
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
    assert resp.get_json() == {'ok': True, 'queued': 4, 'skipped': 1}
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
