"""👁️ The generation queue, made visible (GitHub #44).

The queue has always existed; nothing ever showed it. These tests cover the read
model and the two actions the panel offers, and — the point of the file — they
pin the read model against the DISPATCHER, because a panel that names a job one
thing while `job_queue._dispatch_completion` treats it as another would be worse
than no panel at all.
"""
import json
from datetime import datetime, timedelta

import pytest


def _row(cls, job_id, metadata, *, status='pending', priority=10, minutes=0):
    return cls(job_id=job_id, user_id='local', status=status, priority=priority,
               workflow_data='{}',
               created_at=datetime(2026, 1, 1, 12, 0) + timedelta(minutes=minutes),
               job_metadata=json.dumps(metadata))


def _add(db, *rows):
    for row in rows:
        db.session.add(row)
    db.session.commit()


# --- what a job is ------------------------------------------------------------

@pytest.mark.parametrize('metadata,title,surface', [
    ({'is_lora_test': True, 'dataset_id': 3}, 'Test Studio image', '🧪 Test Studio'),
    ({'is_lora_test': True, 'derivation_kind': 'canvas_image_improve'},
     'Upscale & improve', '◉ Canvas'),
    ({'is_reference_edit': True, 'model_name': 'klein_edit_dataset'},
     'Reference edit', '✦ Edit reference'),
    ({'is_bank_improve': True}, 'Upscale & improve', '🗃️ Bank'),
    ({'model_name': 'watermark_klein'}, 'Watermark inpaint', '🧽 Clean watermarks'),
    ({'model_name': 'klein_edit_dataset', 'action': 'upscale_improve',
      'improve_engine': 'seedvr2'}, 'Upscale & improve', '📁 Dataset'),
    ({'model_name': 'klein_edit_dataset'}, 'Generation', '📁 Dataset'),
])
def test_every_kind_of_job_names_itself(app, metadata, title, surface):
    from app.models import ImageGenerationQueue
    from app.services import queue_view
    with app.app_context():
        job = queue_view.describe(_row(ImageGenerationQueue, 'j', metadata))
    assert (job['title'], job['surface']) == (title, surface)


def test_the_panel_reads_the_same_keys_in_the_same_order_as_the_dispatcher(app):
    """The contract. `_dispatch_completion` routes on is_lora_test, then
    is_reference_edit, then is_bank_improve, then model_name — in that order,
    because several of those keys travel TOGETHER (a reference edit rides the
    Klein helper, so it carries its model_name too). `describe` must break the
    same ties the same way, or the panel lies about what is running."""
    import inspect

    from app import job_queue
    from app.models import ImageGenerationQueue
    from app.services import queue_view

    source = inspect.getsource(job_queue._dispatch_completion)
    order = [key for key in ('is_lora_test', 'is_reference_edit', 'is_bank_improve')
             if f"md.get('{key}')" in source]
    assert order == ['is_lora_test', 'is_reference_edit', 'is_bank_improve'], \
        'the dispatcher changed its routing order — describe() must follow it'

    # The tie the order actually decides, exercised rather than asserted on text.
    with app.app_context():
        job = queue_view.describe(_row(ImageGenerationQueue, 'j', {
            'is_reference_edit': True, 'model_name': 'klein_edit_dataset',
            'action': 'upscale_improve'}))
    assert job['title'] == 'Reference edit'


def test_the_canvas_derivation_name_is_the_one_the_studio_stores(app):
    """`queue_view` must not import the studio (it sits upstream of it), so the
    derivation key is written out by hand there. This is the pin that keeps the
    two spellings from drifting apart."""
    from app.services import lora_test_studio, queue_view
    assert lora_test_studio.CANVAS_IMAGE_IMPROVE == 'canvas_image_improve'
    from app.models import ImageGenerationQueue
    with app.app_context():
        job = queue_view.describe(_row(ImageGenerationQueue, 'j', {
            'is_lora_test': True,
            'derivation_kind': lora_test_studio.CANVAS_IMAGE_IMPROVE}))
    assert job['surface'] == '◉ Canvas'


# --- what the panel may touch -------------------------------------------------

def test_a_pass_blocked_on_its_own_job_keeps_it(app):
    """The watermark inpaint and the reference edit are waited on synchronously.
    They are shown — seeing why the GPU is busy is the point — but cancelling
    them from here would leave that pass waiting on a result that never comes."""
    from app.models import ImageGenerationQueue
    from app.services import queue_view
    with app.app_context():
        for metadata in ({'model_name': 'watermark_klein'},
                         {'model_name': 'watermark_klein_mask'},
                         {'is_reference_edit': True}):
            job = queue_view.describe(_row(ImageGenerationQueue, 'j', metadata))
            assert job['cancellable'] is False
            assert job['blocked_by'], 'a refusal must name where the real Stop is'


def test_ordinary_generations_stay_cancellable(app):
    from app.models import ImageGenerationQueue
    from app.services import queue_view
    with app.app_context():
        for metadata in ({'model_name': 'klein_edit_dataset'},
                         {'is_lora_test': True},
                         {'is_bank_improve': True}):
            assert queue_view.describe(
                _row(ImageGenerationQueue, 'j', metadata))['cancellable'] is True


# --- the listing --------------------------------------------------------------

def test_the_listing_opens_on_what_is_running_then_the_line_behind_it(app):
    """Two rules, in this order.

    What is on the GPU comes FIRST: the worker order alone would have placed it
    wherever its created_at fell — in the middle of the very jobs it is holding
    up — which reads as a broken sort rather than as the answer to "what is the
    GPU doing right now?". Behind it, the WAIT is in the worker's own order
    (`priority DESC, created_at ASC`, the same ORDER BY job_queue claims with).

    And only what still owes GPU time: a finished job is history, not a queue."""
    from app.extensions import db
    from app.models import ImageGenerationQueue
    from app.services import queue_view
    meta = {'model_name': 'klein_edit_dataset'}
    with app.app_context():
        _add(db,
             _row(ImageGenerationQueue, 'old', meta, minutes=0),
             _row(ImageGenerationQueue, 'new', meta, minutes=5),
             _row(ImageGenerationQueue, 'promoted', meta, minutes=9, priority=20),
             _row(ImageGenerationQueue, 'running', meta, minutes=1, status='processing'),
             _row(ImageGenerationQueue, 'done', meta, minutes=2, status='completed'))
        listing = queue_view.list_queue()

    assert [j['job_id'] for j in listing['jobs']] == \
        ['running', 'promoted', 'old', 'new']
    assert listing == {**listing, 'queued': 3, 'generating': 1, 'stalled': 0}
    # Positions number the WAIT, so the job on the GPU does not take a place in
    # a line it has already left.
    assert {j['job_id']: j['position'] for j in listing['jobs']} == \
        {'running': 0, 'promoted': 1, 'old': 2, 'new': 3}


def test_a_promoted_job_says_so(app):
    from app.extensions import db
    from app.models import ImageGenerationQueue
    from app.services import queue_view
    meta = {'model_name': 'klein_edit_dataset'}
    with app.app_context():
        _add(db, _row(ImageGenerationQueue, 'a', meta),
             _row(ImageGenerationQueue, 'b', meta, minutes=1))
        assert queue_view.promote('b') == {'ok': True}
        listing = queue_view.list_queue()

    assert [j['job_id'] for j in listing['jobs']] == ['b', 'a']
    assert [j['promoted'] for j in listing['jobs']] == [True, False]


def test_promoting_twice_keeps_run_next_meaning_next(app):
    from app.extensions import db
    from app.models import ImageGenerationQueue
    from app.services import queue_view
    meta = {'model_name': 'klein_edit_dataset'}
    with app.app_context():
        _add(db, _row(ImageGenerationQueue, 'a', meta),
             _row(ImageGenerationQueue, 'b', meta, minutes=1),
             _row(ImageGenerationQueue, 'c', meta, minutes=2))
        queue_view.promote('b')
        queue_view.promote('c')
        order = [j['job_id'] for j in queue_view.list_queue()['jobs']]
    assert order == ['c', 'b', 'a']


def test_a_job_already_on_the_gpu_cannot_be_reordered(app):
    """Nothing to re-order: the worker took it. Refused out loud rather than
    accepted into a no-op, which is the shape of a button that lies."""
    from app.extensions import db
    from app.models import ImageGenerationQueue
    from app.services import queue_view
    with app.app_context():
        _add(db, _row(ImageGenerationQueue, 'running',
                      {'model_name': 'klein_edit_dataset'}, status='processing'))
        result = queue_view.promote('running')
        assert queue_view.describe(
            ImageGenerationQueue.query.filter_by(job_id='running').first()
        )['promotable'] is False
    assert result['ok'] is False and result['status'] == 409


def test_an_unknown_job_is_a_404_not_a_crash(app):
    from app.services import queue_view
    with app.app_context():
        assert queue_view.promote('nope')['status'] == 404


def test_unreadable_metadata_still_lists_the_job(app):
    """A row whose blob cannot be parsed is still occupying the GPU. Hiding it
    would make the queue view lie by omission."""
    from app.extensions import db
    from app.models import ImageGenerationQueue
    from app.services import queue_view
    with app.app_context():
        db.session.add(ImageGenerationQueue(
            job_id='broken', user_id='local', status='pending', workflow_data='{}',
            job_metadata='{not json'))
        db.session.commit()
        jobs = queue_view.list_queue()['jobs']
    assert [j['job_id'] for j in jobs] == ['broken']
    assert jobs[0]['title'] == 'Generation'


# --- the routes ---------------------------------------------------------------

def test_the_route_names_the_dataset_each_job_belongs_to(app, client):
    from app.config import LOCAL_USER
    from app.extensions import db
    from app.models import ImageGenerationQueue
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Queue names', 'queue')
        _add(db, _row(ImageGenerationQueue, 'j',
                      {'model_name': 'klein_edit_dataset', 'dataset_id': ds.id}))
        expected = ds.name
    body = client.get('/api/system/queue').get_json()
    assert body['ok'] and body['jobs'][0]['dataset_name'] == expected


def test_the_route_says_when_the_whole_queue_is_held_from_outside(app, client):
    """Training and the vision pass hold the GPU OUTSIDE this queue — the worker
    claims nothing while either runs. A listing that counted a line going nowhere
    and said nothing about why would be #44 rebuilt one level up."""
    from app.extensions import db
    from app.job_queue import queue_manager
    from app.models import ImageGenerationQueue
    with app.app_context():
        _add(db, _row(ImageGenerationQueue, 'waiting', {'model_name': 'klein_edit_dataset'}))
        assert client.get('/api/system/queue').get_json()['paused_reason'] is None
        queue_manager._set_system_state('training_in_progress', True)
    body = client.get('/api/system/queue').get_json()
    assert body['queued'] == 1
    assert 'training' in (body['paused_reason'] or '').lower()


def test_cancelling_a_pass_owned_job_is_refused_with_its_owner(app, client):
    from app.extensions import db
    from app.models import ImageGenerationQueue
    with app.app_context():
        _add(db, _row(ImageGenerationQueue, 'wm', {'model_name': 'watermark_klein'}))
    response = client.post('/api/system/queue/wm/cancel')
    assert response.status_code == 409
    assert 'Clean watermarks' in response.get_json()['error']
    with app.app_context():
        assert ImageGenerationQueue.query.filter_by(job_id='wm').first().status == 'pending'


def test_cancelling_a_queued_generation_cancels_it(app, client):
    from app.extensions import db
    from app.models import ImageGenerationQueue
    with app.app_context():
        _add(db, _row(ImageGenerationQueue, 'gen', {'model_name': 'klein_edit_dataset'}))
    assert client.post('/api/system/queue/gen/cancel').status_code == 200
    with app.app_context():
        assert ImageGenerationQueue.query.filter_by(job_id='gen').first().status == 'cancelled'
        from app.services import queue_view
        assert queue_view.list_queue()['jobs'] == []


def test_cancelling_something_already_gone_says_so(app, client):
    assert client.post('/api/system/queue/ghost/cancel').status_code == 404
