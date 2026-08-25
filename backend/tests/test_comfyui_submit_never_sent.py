"""GitHub #51 (charlesangus): the recurring "paused ComfyUI job" barrier.

Two defects, one report. The /prompt POST carried a flat 10 s timeout, and
ComfyUI validates the whole graph synchronously on the same event loop its
executor blocks — so a ComfyUI busy loading a 9 GB model on a VRAM-starved
card regularly took longer than 10 s to ANSWER, and every such wait became
the human-confirm recovery barrier ("restart does not seem to help", because
the next attempt timed out identically). And every submit failure — including
ones that provably never reached ComfyUI at all, like a refused connection —
flowed into that same barrier, though no remote prompt can exist when no TCP
session ever did.

Pinned here: the read budget, and the never-sent classification that turns a
provably-unsubmitted failure into an ordinary terminal fail the user simply
retries, while every genuinely ambiguous outcome keeps the strict barrier.
"""
from unittest.mock import patch

import pytest
import requests

from app.utils import comfyui


def _wf():
    return {'1': {'class_type': 'KSampler', 'inputs': {}}}


def _quiet_preflights(monkeypatch):
    monkeypatch.setattr(comfyui, '_ensure_comfyui_before_generation', lambda: (True, 'up'))
    monkeypatch.setattr(comfyui, 'fetch_object_info_model_files', lambda: None)
    monkeypatch.setattr(comfyui, 'unsupported_enum_values', lambda wf: [])
    monkeypatch.setattr(comfyui, 'unavailable_model_files', lambda wf: [])


def _refused():
    """A requests ConnectionError shaped like a real connection-refused: its
    args[0] is urllib3's MaxRetryError whose .reason is a NewConnectionError.
    Built by name so the test needs no urllib3 import, exactly like the code."""
    reason = type('NewConnectionError', (), {})()
    retry = type('MaxRetryError', (), {'reason': reason})()
    return requests.exceptions.ConnectionError(retry)


# --- the classifier -------------------------------------------------------------

def test_never_sent_truth_table():
    assert comfyui._request_never_sent(requests.exceptions.ConnectTimeout('t')) is True
    assert comfyui._request_never_sent(_refused()) is True
    # A read timeout postdates an accepted POST: ambiguous, keeps the barrier.
    assert comfyui._request_never_sent(requests.exceptions.ReadTimeout('t')) is False
    # A reset mid-response is ambiguous too — its reason is not a NewConnectionError.
    proto = type('ProtocolError', (), {})()
    retry = type('MaxRetryError', (), {'reason': proto})()
    assert comfyui._request_never_sent(
        requests.exceptions.ConnectionError(retry)) is False
    # A bare ConnectionError with no urllib3 envelope proves nothing either.
    assert comfyui._request_never_sent(
        requests.exceptions.ConnectionError('reset by peer')) is False


# --- the tag on the wire --------------------------------------------------------

@pytest.mark.parametrize('exc', [
    requests.exceptions.ConnectTimeout('connect timed out'),
    _refused(),
])
def test_a_connection_that_never_existed_is_tagged_unreachable(app, monkeypatch, exc):
    _quiet_preflights(monkeypatch)

    def boom(*a, **k):
        raise exc
    monkeypatch.setattr(comfyui.requests, 'post', boom)
    with app.app_context():
        result, error = comfyui.queue_prompt_to_comfyui(_wf(), 'client')
    assert result is None
    assert error.startswith('COMFYUI_UNREACHABLE')


def test_a_read_timeout_is_not_tagged_it_stays_ambiguous(app, monkeypatch):
    """ComfyUI may have accepted the POST before going quiet — this MUST keep
    flowing into the unknown-submit barrier, not become a clean fail."""
    _quiet_preflights(monkeypatch)

    def boom(*a, **k):
        raise requests.exceptions.ReadTimeout('read timed out')
    monkeypatch.setattr(comfyui.requests, 'post', boom)
    with app.app_context():
        result, error = comfyui.queue_prompt_to_comfyui(_wf(), 'client')
    assert result is None
    assert not error.startswith('COMFYUI_UNREACHABLE')
    assert not error.startswith('WORKFLOW_INVALIDE')


def test_a_pre_post_refusal_is_tagged_and_never_posts(app, monkeypatch):
    _quiet_preflights(monkeypatch)
    monkeypatch.setattr(comfyui, '_ensure_comfyui_before_generation',
                        lambda: (False, 'ComfyUI is stopped'))
    monkeypatch.setattr(comfyui.requests, 'post',
                        lambda *a, **k: pytest.fail('POSTed with ComfyUI down'))
    with app.app_context():
        result, error = comfyui.queue_prompt_to_comfyui(_wf(), 'client')
    assert result is None
    assert error.startswith('COMFYUI_UNREACHABLE')
    assert 'ComfyUI is stopped' in error


def test_the_post_grants_comfyui_a_long_read_budget(app, monkeypatch):
    """The flat timeout=10 was the recurring-barrier machine: /prompt validates
    the graph synchronously, and a busy install sits past 10 s before it
    answers. Connect stays short; only the READ budget is generous."""
    _quiet_preflights(monkeypatch)
    seen = {}

    class _Resp:
        status_code = 200
        def raise_for_status(self):
            return None
        def json(self):
            return {'prompt_id': 'p1'}

    def capture(url, **kw):
        seen.update(kw)
        return _Resp()
    monkeypatch.setattr(comfyui.requests, 'post', capture)
    with app.app_context():
        result, error = comfyui.queue_prompt_to_comfyui(_wf(), 'client')
    assert error is None and result == {'prompt_id': 'p1'}
    connect, read = seen['timeout']
    assert connect <= 10, 'nobody-listening must still fail fast'
    assert read >= 120, 'a busy ComfyUI needs a real answer window'


# --- the queue mapping ----------------------------------------------------------

def test_submit_maps_unreachable_to_terminal_rejection(app):
    from app.job_queue import _ComfySubmitRejected, _submit
    with app.app_context():
        with patch('app.utils.comfyui.queue_prompt_to_comfyui',
                   return_value=(None, 'COMFYUI_UNREACHABLE (nothing was submitted): x')):
            with pytest.raises(_ComfySubmitRejected):
                _submit({'1': {}}, 'client-1')


def test_submit_keeps_ambiguous_errors_on_the_barrier_path(app):
    from app.job_queue import _ComfySubmitUnknown, _submit
    with app.app_context():
        with patch('app.utils.comfyui.queue_prompt_to_comfyui',
                   return_value=(None, 'Failed to connect or communicate with '
                                       'ComfyUI API (http://x): read timed out')):
            with pytest.raises(_ComfySubmitUnknown):
                _submit({'1': {}}, 'client-1')


# --- end to end through the queue ----------------------------------------------

def test_never_sent_fails_the_job_without_raising_the_barrier(app):
    """THE regression #51 describes, from the other side: ComfyUI simply not
    there must leave a failed job the user retries — never a paused-job banner
    demanding a restart nobody needed."""
    from app.job_queue import queue_manager
    from app.models import ImageGenerationQueue
    with app.app_context():
        jid = queue_manager.add_job(workflow_data={'1': {}})
    with patch('app.utils.comfyui.queue_prompt_to_comfyui',
               return_value=(None, 'COMFYUI_UNREACHABLE (nothing was submitted): '
                                   'could not connect')), \
         patch('app.job_queue._dispatch_completion'):
        with app.app_context():
            assert queue_manager.process_one() is True
            row = ImageGenerationQueue.query.filter_by(job_id=jid).one()
            assert row.status == 'failed'
            assert queue_manager.has_comfyui_stalled_barrier() is False


def test_an_ambiguous_submit_still_raises_the_barrier(app):
    """The strictness that must survive this fix: an error that can postdate an
    accepted POST keeps the fail-closed barrier and the stalled row."""
    from app.job_queue import queue_manager
    from app.models import ImageGenerationQueue
    with app.app_context():
        jid = queue_manager.add_job(workflow_data={'1': {}})
    with patch('app.utils.comfyui.queue_prompt_to_comfyui',
               return_value=(None, 'Failed to connect or communicate with '
                                   'ComfyUI API (http://x): read timed out')), \
         patch('app.job_queue._dispatch_completion'):
        with app.app_context():
            queue_manager.process_one()
            row = ImageGenerationQueue.query.filter_by(job_id=jid).one()
            assert row.status == 'stalled'
            assert queue_manager.has_comfyui_stalled_barrier() is True
