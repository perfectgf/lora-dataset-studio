"""The Ollama fence as the user meets it: a refusal that carries its remedy.

Three things have to be true at the HTTP seam for the UI to be able to offer
"unload it and continue" and to wait for the block to clear on its own:
the refusal must be machine-recognisable (a code, not a sentence to match),
the state must be pollable, and the eviction must be impossible without an
explicit confirmation.
"""
from unittest.mock import patch

import pytest

from app.services import ollama_gpu_fence as fence
from app.services.vision_ollama import LocalOllamaFenceError

# Drives the fence directly (see conftest's autouse stub).
pytestmark = pytest.mark.ollama_fence


def test_enhance_refusal_carries_the_fence_code_so_the_ui_can_offer_the_way_out(
        app, client, monkeypatch):
    monkeypatch.setattr('app.services.lora_test_studio.enhance_test_prompt',
                        lambda prompt, model=None: (_ for _ in ()).throw(
                            LocalOllamaFenceError(fence.FENCE_BLOCKED_MESSAGE)))
    res = client.post('/api/studio/enhance-prompt', json={'prompt': 'a girl'})
    body = res.get_json()
    assert res.status_code == 409
    assert body['code'] == 'ollama_fence_blocked'
    # The sentence is still there: the banner shows it verbatim.
    assert 'already in use outside LDS' in body['error']


def test_an_ordinary_enhance_failure_keeps_its_bare_409_without_the_code(
        app, client, monkeypatch):
    """Only the fence earns the button. A missing model must not get one."""
    monkeypatch.setattr('app.services.lora_test_studio.enhance_test_prompt',
                        lambda prompt, model=None: (_ for _ in ()).throw(
                            RuntimeError('Ollama is unavailable')))
    res = client.post('/api/studio/enhance-prompt', json={'prompt': 'a girl'})
    assert res.status_code == 409
    assert 'code' not in res.get_json()


def test_the_fence_state_route_reports_what_is_blocking(app, client):
    endpoint = 'http://127.0.0.1:11434'
    with patch.object(fence, '_configured_local_endpoint', return_value=('local', endpoint)), \
            patch.object(fence.requests, 'get',
                         return_value=type('R', (), {
                             'status_code': 200,
                             'json': lambda self: {'models': [{'name': 'other-app-model'}]},
                         })()):
        body = client.get('/api/system/ollama-fence').get_json()
    assert body['ok'] is True and body['blocked'] is True
    assert body['models'] == ['other-app-model']


def test_the_unload_route_refuses_without_an_explicit_confirmation(app, client):
    with patch.object(fence, 'unload_foreign_models') as unload:
        res = client.post('/api/system/ollama-fence/unload', json={})
        assert res.status_code == 400
        res = client.post('/api/system/ollama-fence/unload',
                          json={'confirmed_unload_external': 'yes'})
        assert res.status_code == 400
    unload.assert_not_called()


def test_the_unload_route_evicts_only_once_the_user_has_said_so(app, client):
    with patch.object(fence, 'unload_foreign_models',
                      return_value={'ok': True, 'reason': 'unloaded',
                                    'unloaded': ['other-app-model'],
                                    'still_loaded': []}) as unload:
        res = client.post('/api/system/ollama-fence/unload',
                          json={'confirmed_unload_external': True})
    assert res.status_code == 200 and res.get_json()['unloaded'] == ['other-app-model']
    unload.assert_called_once_with()


def test_every_unload_sentence_names_the_server_this_install_runs(app, client):
    """Four sentences on this route name the server, and all four said Ollama.

    This is the screen whose entire job is to explain which model is holding the
    card and what evicting it costs. Under LM Studio it named a product the user
    may not have installed -- in the consent prompt itself, the one place the app
    asks permission to touch something it does not own.
    """
    from app import config
    config.save_config({'local_llm': {'provider': 'lmstudio'}})

    refusal = client.post('/api/system/ollama-fence/unload', json={}).get_json()['error']
    assert 'LM Studio' in refusal and 'Ollama' not in refusal

    for reason, expected in (('not-local', 'remote'), ('unreachable', 'did not answer'),
                             ('still-loaded', 'still reports a model')):
        with patch.object(fence, 'unload_foreign_models',
                          return_value={'ok': False, 'reason': reason}):
            res = client.post('/api/system/ollama-fence/unload',
                              json={'confirmed_unload_external': True})
        body = res.get_json()
        assert res.status_code == 409 and expected in body['error']
        assert 'LM Studio' in body['error'], f'{reason} does not name the provider'
        assert 'Ollama' not in body['error'], f'{reason} names the wrong product'


def test_the_unload_sentences_still_say_ollama_where_ollama_is_what_runs(app, client):
    """The other half: renaming must not have made the copy provider-blind."""
    from app import config
    config.save_config({'local_llm': {'provider': 'ollama'}})
    refusal = client.post('/api/system/ollama-fence/unload', json={}).get_json()['error']
    assert 'Ollama' in refusal and 'LM Studio' not in refusal


def test_the_share_route_refuses_without_an_explicit_confirmation(app, client):
    """Sharing one card between two loaded models is not free — on Windows it
    pages instead of failing. So it is a decision, taken once, in words."""
    with patch.object(fence, 'share_gpu_with_foreign_model') as share:
        assert client.post('/api/system/ollama-fence/share', json={}).status_code == 400
        assert client.post('/api/system/ollama-fence/share',
                           json={'confirmed_share_gpu': 'yes'}).status_code == 400
    share.assert_not_called()


def test_the_share_route_answers_the_hold_once_the_user_has_said_so(app, client):
    with patch.object(fence, 'share_gpu_with_foreign_model',
                      return_value={'ok': True, 'reason': 'sharing', 'seconds': 900,
                                    'models': ['other-app-model']}) as share:
        res = client.post('/api/system/ollama-fence/share',
                          json={'confirmed_share_gpu': True})
    assert res.status_code == 200 and res.get_json()['seconds'] == 900
    share.assert_called_once_with()


def test_sharing_when_nothing_is_held_says_so_instead_of_pretending(app, client):
    with patch.object(fence, 'share_gpu_with_foreign_model',
                      return_value={'ok': False, 'reason': 'not-blocked', 'seconds': 0}):
        res = client.post('/api/system/ollama-fence/share',
                          json={'confirmed_share_gpu': True})
    assert res.status_code == 409
    assert 'Nothing is holding the queue' in res.get_json()['error']


def test_a_failed_unload_answers_409_with_a_sentence_the_user_can_act_on(app, client):
    with patch.object(fence, 'unload_foreign_models',
                      return_value={'ok': False, 'reason': 'still-loaded',
                                    'unloaded': [], 'still_loaded': ['busy-model']}):
        res = client.post('/api/system/ollama-fence/unload',
                          json={'confirmed_unload_external': True})
    assert res.status_code == 409
    assert 'still reports a model in memory' in res.get_json()['error']
