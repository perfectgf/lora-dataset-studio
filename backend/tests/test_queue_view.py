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


def test_the_hold_sentence_is_written_for_whoever_is_reading_it(app):
    """The queue is app-wide; its pause has to be too.

    This used to relay `lora_test_studio.gpu_busy_reason()` verbatim, whose
    sentences are written for the Test Studio — so someone in the dataset
    workspace was told "the studio is unavailable", about a screen they were not
    on, and pointed at a paused test they had never opened."""
    from app.job_queue import (HOLD_COMFYUI_RECOVERY, HOLD_LABELS,
                               HOLD_OLLAMA_FENCE, HOLD_TRAINING, HOLD_VISION)
    from app.services import queue_view

    # The keys the worker answers with are the keys this module words. The
    # fence hold is deliberately NOT in the static dict: its sentence is
    # dynamic (_ollama_fence_sentence), worded from what the fence saw.
    assert set(queue_view._HOLD_SENTENCES) == {
        HOLD_TRAINING, HOLD_VISION, HOLD_COMFYUI_RECOVERY}
    assert set(queue_view._HOLD_SENTENCES) | {HOLD_OLLAMA_FENCE} == set(HOLD_LABELS)
    for sentence in queue_view._HOLD_SENTENCES.values():
        assert 'studio' not in sentence.lower(), sentence
        assert sentence.endswith('.')

    with app.app_context():
        assert queue_view.paused_reason() is None


def test_the_in_process_vision_window_is_a_hold_the_dock_can_name(app, monkeypatch):
    """The worker refuses to claim on FOUR conditions; the DB flags carry three.

    The fourth — the in-process vision window — is the one case where the
    heartbeat can lose ownership and the flag expire by TTL while the barrier
    itself is kept. It was therefore the single pause with no explanation
    anywhere in the app, which is exactly the one a queue view owes the user."""
    from app import job_queue
    from app.services import queue_view
    monkeypatch.setattr(job_queue, '_vision_window_blocks_gpu', lambda: True)
    with app.app_context():
        assert job_queue.queue_manager.gpu_hold() == job_queue.HOLD_VISION
        assert 'vision pass' in (queue_view.paused_reason() or '')


def test_the_ollama_fence_refusal_is_a_hold_the_dock_can_name(app, monkeypatch):
    """The fence refuses AFTER a row is claimed, so it sits outside the four
    pre-claim gates — and it was therefore the remaining pause with no
    explanation anywhere in the app: jobs queued forever, no banner, no log a
    user would recognise (found via a KoboldCPP endpoint in the Ollama slot).
    When the squatter self-identifies as KoboldCPP, the remedy is the Ollama
    URL itself — telling that user to 'unload the model' would be a dead end,
    kcpp never unloads."""
    from app import job_queue
    from app.services import ollama_gpu_fence, queue_view
    monkeypatch.setattr(ollama_gpu_fence, 'last_block',
                        lambda max_age_s=15: {'reason': 'foreign',
                                              'endpoint': 'http://127.0.0.1:5001',
                                              'models': ['kcpp-model'],
                                              'families': ['koboldcpp']})
    with app.app_context():
        assert job_queue.queue_manager.gpu_hold() == job_queue.HOLD_OLLAMA_FENCE
        sentence = queue_view.paused_reason() or ''
    assert 'KoboldCPP' in sentence
    assert 'Ollama URL' in sentence


def test_the_fence_sentence_names_the_squatting_model_when_it_is_not_kcpp(app, monkeypatch):
    """An ordinary foreign residency (someone's own Ollama session) is worded
    with the model's NAME, so the user is not sent hunting for it."""
    from app.services import ollama_gpu_fence, queue_view
    monkeypatch.setattr(ollama_gpu_fence, 'last_block',
                        lambda max_age_s=15: {'reason': 'foreign',
                                              'endpoint': 'http://127.0.0.1:11434',
                                              'models': ['llama3:8b'],
                                              'families': []})
    with app.app_context():
        sentence = queue_view.paused_reason() or ''
    assert 'llama3:8b' in sentence
    assert 'Unload it there' in sentence


def _fence_block(held_seconds, models=('llama3:8b',)):
    return lambda max_age_s=15: {'reason': 'foreign', 'endpoint': 'http://127.0.0.1:11434',
                                 'models': list(models), 'families': [],
                                 'held_seconds': held_seconds}


def test_a_short_fence_hold_is_left_alone_to_clear_on_its_own(app, monkeypatch):
    """Most fence holds end by themselves — Ollama drops an idle model after a
    few minutes. Offering to share the card during the first minute would trade
    a self-healing wait for a real, measured slowdown."""
    from app.services import ollama_gpu_fence, queue_view
    monkeypatch.setattr(ollama_gpu_fence, 'last_block', _fence_block(5))
    with app.app_context():
        assert queue_view.paused_action() is None
        assert 'has been waiting' not in (queue_view.paused_reason() or '')


def test_a_hold_that_outlives_the_minute_says_how_long_and_offers_an_answer(app, monkeypatch):
    """The fix for the open-ended freeze: a wall the user can neither see the
    length of nor answer is what made this hold feel like a bug in the queue."""
    from app.services import ollama_gpu_fence, queue_view
    monkeypatch.setattr(ollama_gpu_fence, 'last_block', _fence_block(260))
    with app.app_context():
        sentence = queue_view.paused_reason() or ''
        action = queue_view.paused_action()
    assert 'It has been waiting 4 min.' in sentence
    assert action['kind'] == 'share_gpu' and action['models'] == ['llama3:8b']
    # The offer states the cost. Sharing one card between two loaded models is
    # not free, and on Windows it degrades silently instead of failing.
    assert 'slower' in action['confirm']


def test_only_the_fence_hold_is_answerable_the_others_end_by_themselves(app, monkeypatch):
    """A training run and a vision pass finish on their own, and a paused
    ComfyUI job has its remedy on another screen: none of them earns a button."""
    from app import job_queue
    from app.services import queue_view
    monkeypatch.setattr(job_queue.queue_manager, '_get_system_state',
                        lambda key, default=False: key == 'training_in_progress')
    with app.app_context():
        assert job_queue.queue_manager.gpu_hold() == job_queue.HOLD_TRAINING
        assert queue_view.paused_action() is None


def test_a_cancelled_job_is_not_reported_to_the_user_as_a_failure(app, client):
    """`link_completed_dataset_image` falls back to a message pointing at the
    server log when no reason is given — so a job the user had just cancelled
    with one click came back labelled 'generation failed', sending them to hunt
    a ComfyUI error that never happened."""
    from app.extensions import db
    from app.models import ImageGenerationQueue
    from app.routes.system import CANCELLED_FROM_QUEUE
    with app.app_context():
        _add(db, _row(ImageGenerationQueue, 'gen', {'model_name': 'klein_edit_dataset'}))
    assert client.post('/api/system/queue/gen/cancel').status_code == 200
    with app.app_context():
        row = ImageGenerationQueue.query.filter_by(job_id='gen').first()
        assert row.status == 'cancelled'
        assert row.error_message == CANCELLED_FROM_QUEUE
        assert 'fail' not in row.error_message.lower()


def test_every_refusal_these_routes_send_is_a_sentence(app, client):
    """A route that answers a person does not hand back an internal string. The
    404 of "run next" used to toast the literal words 'not found'."""
    from app.extensions import db
    from app.models import ImageGenerationQueue
    with app.app_context():
        _add(db, _row(ImageGenerationQueue, 'running',
                      {'model_name': 'klein_edit_dataset'}, status='processing'))
    for response in (client.post('/api/system/queue/ghost/next'),
                     client.post('/api/system/queue/running/next'),
                     client.post('/api/system/queue/ghost/cancel')):
        error = response.get_json()['error']
        assert error[0].isupper(), error
        assert error.endswith('.') or '—' in error, error
        assert len(error.split()) >= 4, error


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
