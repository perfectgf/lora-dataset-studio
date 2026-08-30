"""Tests written against a MUTATION, not against the code as it stands.

A verification pass mutated this branch's own fixes out of the tree, ran the
suite, and watched it stay green: the provider-aware dials could be reverted to
hard-coded `ollama.*` keys (103 passed), the JSON mode could be removed from both
sides (57 passed), the whole URL validation could be deleted (57 passed). Those
fixes were guarded by nothing. A test that certifies without proving is worse than
no test at all, because it is believed.

So each test below names the mutation it refuses, and asserts the OBSERVABLE
consequence rather than the shape of the code. If a future reader deletes the fix,
this file is what turns red.
"""
import errno
import json

import pytest

from app import config
from app.services import ollama_gpu_fence as fence
from app.services import vision_lmstudio as lms
from app.services import vision_keepalive, vision_llm, vision_pool


# --- MUTATION: read `ollama.vision_concurrency` / `ollama.vision_keep_warm_seconds`
#     unconditionally, ignoring the provider. Was green.

@pytest.mark.parametrize('provider, concurrency, warm', [
    ('ollama', 2, 60),
    ('lmstudio', 8, 300),
])
def test_the_two_dials_read_the_active_providers_own_keys(app, provider, concurrency, warm):
    """The values differ per provider ON PURPOSE here, so a hard-coded `ollama.`
    key cannot produce the LM Studio expectation by accident. The previous test
    set both sections to the same numbers, which is why the mutation survived."""
    with app.app_context():
        config.save_config({
            'local_llm': {'provider': provider},
            'ollama': {'vision_concurrency': 2, 'vision_keep_warm_seconds': 60},
            'lmstudio': {'vision_concurrency': 8, 'vision_keep_warm_seconds': 300},
        })
        assert vision_pool.vision_concurrency() == concurrency
        assert vision_keepalive.warm_seconds() == warm


# --- MUTATION: drop `response_format` from _chat AND the fmt→as_json translation
#     in the router. Was green — while head-crop and watermark bbox PARSE the answer.

def test_a_json_request_really_leaves_as_a_json_request(app, monkeypatch):
    sent = {}

    def _post(url, *a, **kw):
        sent.update(kw.get('json') or {})
        return _ok_chat()

    monkeypatch.setattr(lms.requests, 'get', _one_vlm())
    monkeypatch.setattr(lms.requests, 'post', _post)
    monkeypatch.setattr(lms, '_data_uri_ok', None)
    with app.app_context():
        config.save_config({'local_llm': {'provider': 'lmstudio'},
                            'lmstudio': {'url': 'http://127.0.0.1:1299'}})
        vision_llm.describe_image(_jpeg(), 'bbox please', fmt='json')
    rf = sent.get('response_format') or {}
    assert rf, ('the caller asked for JSON and the request went out as free text — '
               'the head-crop and watermark parsers read this answer')
    # The PROPERTY, not one spelling. This assertion used to pin
    # {'type': 'json_object'} — correct in intent, and the exact value LM Studio
    # 0.4.23 rejects with HTTP 400 "'response_format.type' must be 'json_schema'
    # or 'text'". It was green while framing classify was down on every call,
    # because a unit test cannot know which spelling a server accepts. What the
    # driver owes is a JSON mode that a supported build honours.
    assert rf.get('type') in ('json_schema', 'json_object')
    if rf['type'] == 'json_schema':
        # Permissive on purpose: the callers ask for an object, never a named
        # schema, and LM Studio enforces the grammar from it.
        assert (rf.get('json_schema') or {}).get('schema') == {'type': 'object'}


def test_a_rejected_json_spelling_is_retried_with_the_other_one(app, monkeypatch):
    """Which spelling works is a per-BUILD fact, so neither may be hard-coded.

    Measured on 0.4.23: the historical OpenAI `json_object` is refused outright.
    Older builds accept it. A driver that picked one and stopped would be broken
    on half the versions, silently, only on the passes that parse JSON.
    """
    seen = []

    class _Refused:
        status_code = 400
        text = """{"error": "'response_format.type' must be 'json_schema' or 'text'"}"""

    def _post(url, *a, **kw):
        body = kw.get('json') or {}
        kind = (body.get('response_format') or {}).get('type')
        seen.append(kind)
        return _ok_chat() if kind == 'json_object' else _Refused()

    monkeypatch.setattr(lms.requests, 'get', _one_vlm())
    monkeypatch.setattr(lms.requests, 'post', _post)
    monkeypatch.setattr(lms, '_data_uri_ok', None)
    monkeypatch.setattr(lms, '_json_format', None)
    with app.app_context():
        config.save_config({'local_llm': {'provider': 'lmstudio'},
                            'lmstudio': {'url': 'http://127.0.0.1:1299'}})
        assert vision_llm.describe_image(_jpeg(), 'bbox please', fmt='json')
    assert seen == ['json_schema', 'json_object'], (
        'the refused spelling was not retried with the other one')
    # ...and the working one is remembered, so the next call pays one request.
    assert lms._json_format == 'json_object'


def test_a_plain_request_does_not_ask_for_json(app, monkeypatch):
    """The other half of the same claim: the flag must TRAVEL, not be always-on."""
    sent = {}
    monkeypatch.setattr(lms.requests, 'get', _one_vlm())
    monkeypatch.setattr(lms.requests, 'post',
                        lambda url, *a, **kw: sent.update(kw.get('json') or {}) or _ok_chat())
    monkeypatch.setattr(lms, '_data_uri_ok', None)
    with app.app_context():
        config.save_config({'local_llm': {'provider': 'lmstudio'},
                            'lmstudio': {'url': 'http://127.0.0.1:1299'}})
        vision_llm.describe_image(_jpeg(), 'describe')
    assert 'response_format' not in sent


# --- MUTATION: delete the scheme/credentials/port refusals from _suffix_free.
#     Was green — while a password typed into the URL came back out of the API.

@pytest.mark.parametrize('bad', [
    'http://user:hunter2@127.0.0.1:1234',      # credentials must never survive
    'ftp://127.0.0.1:1234',                    # not an HTTP origin
    'http://127.0.0.1:notaport',               # malformed port
    '127.0.0.1:1234',                          # no scheme at all
])
def test_an_unusable_url_is_refused_rather_than_partly_honoured(bad):
    assert lms._suffix_free(bad) == '', f'{bad!r} was accepted'


def test_a_typed_password_never_reaches_the_capabilities_payload(app, client):
    """The consequence, not the mechanism: whatever the driver does internally,
    a credential typed into the URL must not come back out of the API."""
    with app.app_context():
        config.save_config({'local_llm': {'provider': 'lmstudio'},
                            'lmstudio': {'url': 'http://user:hunter2@127.0.0.1:1299'}})
    body = json.dumps(client.get('/api/capabilities').get_json())
    assert 'hunter2' not in body


# --- MUTATION: ignore _borrowed_models in _release_endpoint. Was green, because
#     the existing test reached the refusal through the `foreign` branch instead.

@pytest.mark.ollama_fence      # drives the REAL _probe; the autouse stub would hide it
def test_a_borrowed_model_is_refused_through_the_borrowed_branch(app, monkeypatch):
    """The endpoint is deliberately NOT marked foreign here — admission clears it
    for a borrowed model — so the only thing that can refuse the hand-off is the
    borrowed check itself. Remove it and this goes red."""
    monkeypatch.setattr(lms, 'probe_resident',
                        lambda ep: ('models', ['qwen/qwen3-vl-4b'], {}))
    monkeypatch.setattr(lms, 'release',
                        lambda *a, **kw: pytest.fail('a borrowed model was unloaded'))
    fence._owned_models.clear(); fence._borrowed_models.clear()
    fence._endpoint_driver.clear(); fence._foreign_local_endpoints.clear()
    url = 'http://127.0.0.1:1299'
    with app.app_context():
        config.save_config({'local_llm': {'provider': 'lmstudio'},
                            'lmstudio': {'url': url}})
        assert fence.mark_before_generate(url, 'qwen/qwen3-vl-4b',
                                          provider='lmstudio') == 'local'
        assert url not in fence._foreign_local_endpoints, 'precondition: not foreign'
        assert fence._release_endpoint(url, set()) is False
    fence._owned_models.clear(); fence._borrowed_models.clear()
    fence._endpoint_driver.clear(); fence._foreign_local_endpoints.clear()


# --- MUTATION: hard-code 'ollama' as the diagnostic's provider. Was green.

def test_the_diagnostic_reports_the_provider_that_is_actually_configured(app, client):
    with app.app_context():
        config.save_config({'local_llm': {'provider': 'lmstudio'}})
    assert client.get('/api/diagnostic').get_json()['local_llm']['provider'] == 'lmstudio'
    with app.app_context():
        config.save_config({'local_llm': {'provider': 'ollama'}})
    assert client.get('/api/diagnostic').get_json()['local_llm']['provider'] == 'ollama'


# --- MUTATION: drop 1234 from the suite's hermeticity guard. Untested until now,
#     so a developer running LM Studio would silently get a different suite.

def test_the_suite_refuses_both_local_llm_daemons(monkeypatch):
    """Port-parsed, not substring: ':1234' also appears in ':12345', an ordinary
    fixture port that must keep working."""
    import requests as rq
    for url in ('http://127.0.0.1:11434/api/tags', 'http://127.0.0.1:1234/api/v1/models'):
        with pytest.raises(rq.exceptions.ConnectionError) as exc:
            rq.get(url, timeout=1)
        assert isinstance(exc.value.args[0], ConnectionRefusedError)
        assert exc.value.args[0].errno == errno.ECONNREFUSED
    # …and a neighbouring port is NOT swallowed by the guard.
    assert getattr(rq.get, 'lds_ollama_guard', False) is True
    try:
        rq.get('http://127.0.0.1:12345/', timeout=0.2)
    except rq.exceptions.ConnectionError as exc:
        inner = exc.args[0] if exc.args else None
        assert 'must not reach' not in str(inner), (
            'the guard swallowed port 12345 — it is matching a substring again')
    except Exception:
        pass


# --- MUTATION: read the LM Studio token from config again. Untested until now.

def test_the_token_is_sent_as_a_bearer_and_never_comes_back_out(app, client, monkeypatch):
    from app.config import set_secrets
    with app.app_context():
        config.save_config({'local_llm': {'provider': 'lmstudio'},
                            'lmstudio': {'url': 'http://127.0.0.1:1299'}})
        set_secrets({'LMSTUDIO_API_KEY': 'lms-tok-DEADBEEF'})
        seen = {}
        monkeypatch.setattr(lms.requests, 'get',
                            lambda url, *a, **kw: seen.update(kw.get('headers') or {})
                            or _Resp404())
        lms.list_models()
        assert seen.get('Authorization') == 'Bearer lms-tok-DEADBEEF'
    # The consequence that matters: it is a SECRET, so it never rides in a payload
    # a user can paste into a bug report.
    assert 'DEADBEEF' not in json.dumps(client.get('/api/settings').get_json())
    assert 'DEADBEEF' not in json.dumps(client.get('/api/diagnostic').get_json())


# --- helpers -----------------------------------------------------------------

class _Resp404:
    status_code = 404
    text = 'not found'

    def json(self):
        raise ValueError('no json')


def _ok_chat():
    class _R:
        status_code = 200
        text = ''

        def json(self):
            return {'choices': [{'message': {'content': 'ok'}}]}
    return _R()


def _one_vlm():
    body = {'models': [{'key': 'qwen/qwen3-vl-4b', 'type': 'vlm',
                        'loaded_instances': [{'id': 'qwen/qwen3-vl-4b'}]}]}

    def _get(url, *a, **kw):
        from urllib.parse import urlsplit

        class _R:
            status_code = 200 if urlsplit(url).path == '/api/v1/models' else 404
            text = ''

            def json(self_inner):
                if self_inner.status_code != 200:
                    raise ValueError('no json')
                return body
        return _R()
    return _get


def _jpeg() -> bytes:
    import io as _io

    from PIL import Image
    buf = _io.BytesIO()
    Image.new('RGB', (48, 48), (200, 30, 30)).save(buf, 'JPEG')
    return buf.getvalue()
