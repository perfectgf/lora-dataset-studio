"""The ChatGPT engine's two lanes must answer the same event the same way.

NOTHING here talks to OpenAI. `requests.post` is patched in every test, no key
is read, no subscription token is minted, no quota is spent — which is also this
file's honest limit: it proves how we READ an answer we hand ourselves, not that
OpenAI produces that shape today. The subscription lane in particular rides an
undocumented endpoint, so its exact bodies are inferred, not observed.

WHAT IS BEING LOCKED
--------------------
`chatgpt_image.py` carries two lanes to the same provider: an API key
(/images/edits) and a ChatGPT subscription (the Codex responses backend). They
disagreed about what a failure is. MEASURED on the tree before this file
existed, with the calls below:

    subscription  HTTP 500          -> None, no exception
    subscription  HTTP 403 / 404    -> None, no exception
    subscription  dropped connection-> None, no exception
    subscription  read timeout      -> None, no exception
    API key       HTTP 500          -> raises, cause named
    API key       dropped connection-> raises, cause named

Every one of those Nones reached the user as the fan-out's sentence for "the
provider answered and produced no image" — the words for a refusal, printed over
a network outage. So, in order of how badly each hurts when broken:

  1. a breakdown on the subscription lane names itself (network, timeout, 5xx)
     and can never read as a content refusal;
  2. the four causes stay four different sentences: refusal / network / quota /
     token — plus the fifth this lane owns, OpenAI closing the door on it;
  3. no message promises a workaround (see the wording test): whether the same
     call would pass again is exactly what we do not know;
  4. the API-key lane, which already worked, is unchanged where it worked — and
     tells apart the two kinds of 400 it used to merge;
  5. what is genuinely unreadable stays None, and says so, rather than being
     given an invented cause.
"""
import base64
import json
import logging
from unittest.mock import MagicMock, patch

import pytest

import requests

KEY = 'sk-proj-TESTKEYVALUE0123456789abcdefghijklmn'
TOKEN = 'oauth-access-TESTTOKEN-0123456789'
PNG = b'\x89PNG\r\n\x1a\nfake-pixels'


# --- fixtures ---------------------------------------------------------------

def _resp(status=200, json_body=None, text='', headers=None):
    r = MagicMock(status_code=status, headers=headers if headers is not None else {})
    if json_body is None:
        r.json.side_effect = ValueError('no json')
    else:
        r.json.return_value = json_body
    r.text = text if text or json_body is None else json.dumps(json_body)
    return r


def _openai_err(message, code=None, param=None, status=400):
    err = {'message': message}
    if code:
        err['code'] = code
    if param:
        err['param'] = param
    return _resp(status, {'error': err})


@pytest.fixture(autouse=True)
def _subscription_catalog(monkeypatch):
    # Refusal fixtures cover generation responses, independently of discovery.
    from lds_api_engines import chatgpt_models
    monkeypatch.setattr(chatgpt_models, 'available_models',
                        lambda **kwargs: [{'id': 'fixture-image-router'}])


def _sub_connected(monkeypatch):
    from lds_api_engines import chatgpt_oauth
    monkeypatch.setattr(chatgpt_oauth, 'access_token', lambda force_refresh=False: TOKEN)
    monkeypatch.setattr(chatgpt_oauth, 'account_id', lambda: 'acc-x')
    monkeypatch.setattr(chatgpt_oauth, 'status',
                        lambda: {'connected': True, 'email': None, 'plan': None})


def _sse(*events):
    return ''.join(f'data: {json.dumps(e)}\n\n' for e in events) + 'data: [DONE]\n'


def _sub_image_sse():
    return _sse({'type': 'response.output_item.done',
                 'item': {'type': 'image_generation_call',
                          'result': base64.b64encode(PNG).decode()}},
                {'type': 'response.completed', 'response': {'output': []}})


def _sub(monkeypatch, **kw):
    """Call the subscription lane with requests.post patched as told."""
    _sub_connected(monkeypatch)
    from lds_api_engines import chatgpt_image
    with patch('lds_api_engines.chatgpt_image.requests.post', **kw):
        return chatgpt_image.generate_variation(b'ref', 'a portrait',
                                                force_lane='subscription')


# --- 1. THE regression: a breakdown is no longer a blank shrug ---------------

@pytest.mark.parametrize('status,body,expected', [
    (500, 'internal error', 'HTTP 500'),
    (502, 'bad gateway', 'HTTP 502'),
    (503, 'overloaded', 'HTTP 503'),
])
def test_a_subscription_5xx_raises_instead_of_returning_none(app, monkeypatch, status,
                                                             body, expected):
    """MEASURED before the fix: `None`, silently, for every one of these. The
    fan-out then wrote the sentence reserved for a provider that answered."""
    from lds_api_engines import chatgpt_image
    from app.services.engine_errors import EngineFatal, EngineRefused
    with pytest.raises(chatgpt_image.ChatGPTImageError) as e:
        _sub(monkeypatch, return_value=_resp(status, text=body))
    msg = str(e.value)
    assert expected in msg
    assert 'your prompt was not refused' in msg
    assert not isinstance(e.value, EngineRefused)      # an outage is not a refusal
    assert not isinstance(e.value, EngineFatal)        # ...and does not stop the run


@pytest.mark.parametrize('exc', [
    requests.ConnectionError('connection aborted'),
    requests.ReadTimeout('read timed out'),
    requests.exceptions.ChunkedEncodingError('connection broken mid-stream'),
])
def test_a_dropped_connection_on_the_subscription_lane_says_so(app, monkeypatch, exc):
    """MEASURED before the fix: `None`. A user whose network died was told they
    had written something the provider would not draw."""
    from lds_api_engines import chatgpt_image
    from app.services.engine_errors import EngineRefused
    with pytest.raises(chatgpt_image.ChatGPTImageError) as e:
        _sub(monkeypatch, side_effect=exc)
    assert 'could not reach ChatGPT' in str(e.value)
    assert not isinstance(e.value, EngineRefused)


# --- 2. one sentence per cause, and never a borrowed one --------------------

def test_the_five_causes_get_five_different_sentences(app, monkeypatch):
    """Refusal, network, quota, token, closed door. If any two of these ever
    collapse into one message, someone is told to fix the wrong thing."""
    from lds_api_engines import chatgpt_image
    said = {}

    def _catch(key, **kw):
        try:
            _sub(monkeypatch, **kw)
        except Exception as e:                          # noqa: BLE001 — that's the point
            said[key] = (type(e).__name__, str(e))

    _catch('refusal', return_value=_sse_refusal_resp())
    _catch('network', side_effect=requests.ConnectionError('name resolution failed'))
    _catch('quota', return_value=_resp(429, text='usage limit reached'))
    _catch('closed', return_value=_resp(403, text='forbidden'))
    # token: connected, but the refreshed token is refused too
    _sub_connected(monkeypatch)
    with patch('lds_api_engines.chatgpt_image.requests.post',
               return_value=_resp(401, text='token expired')):
        try:
            chatgpt_image.generate_variation(b'r', 'p', force_lane='subscription')
        except Exception as e:                          # noqa: BLE001
            said['token'] = (type(e).__name__, str(e))

    assert set(said) == {'refusal', 'network', 'quota', 'closed', 'token'}
    messages = [m for _, m in said.values()]
    assert len(set(messages)) == 5                      # five causes, five sentences
    assert 'text instead of an image' in said['refusal'][1]
    assert 'could not reach ChatGPT' in said['network'][1]
    assert 'quota' in said['quota'][1]
    assert 'reconnect' in said['token'][1].lower()
    assert 'no longer serving' in said['closed'][1]
    # ...and each is carried by the class its caller acts on
    assert said['quota'][0] == 'SubscriptionQuotaExceeded'
    assert said['token'][0] == 'SubscriptionUnavailable'
    assert said['refusal'][0] == 'ChatGPTImageRefused'


def _sse_refusal_resp():
    r = MagicMock(status_code=200, headers={})
    r.text = _sse({'type': 'response.output_item.done',
                   'item': {'type': 'message', 'content': [
                       {'type': 'output_text',
                        'text': "I can't create that image."}]}},
                  {'type': 'response.completed', 'response': {'output': []}})
    return r


def test_a_lane_openai_has_closed_says_that_instead_of_unknown_error(app, monkeypatch):
    """The most useful thing this lane can ever report. It rides an undocumented
    endpoint the module itself warns may be withdrawn; when that happens, every
    user would otherwise go hunting their own settings for a fault not there."""
    from lds_api_engines import chatgpt_image
    from app.services.engine_errors import EngineFatal
    for status in (403, 404, 410):
        with pytest.raises(chatgpt_image.ChatGPTImageFatal) as e:
            _sub(monkeypatch, return_value=_resp(status, text='gone'))
        msg = str(e.value)
        assert 'no longer serving' in msg
        assert 'experimental' in msg
        assert 'API key in Settings' in msg             # the way to keep working
        assert isinstance(e.value, EngineFatal)         # stops the run, once


def test_the_model_answering_in_prose_relays_its_own_words(app, monkeypatch):
    """A 200 that carries a sentence instead of pixels IS a stated reason.
    Paraphrasing it would be a second guess on top of the one being removed."""
    from lds_api_engines import chatgpt_image
    from app.services.engine_errors import EngineFatal, EngineRefused
    with pytest.raises(chatgpt_image.ChatGPTImageRefused) as e:
        _sub(monkeypatch, return_value=_sse_refusal_resp())
    assert "I can't create that image." in str(e.value)
    assert isinstance(e.value, EngineRefused)
    assert not isinstance(e.value, EngineFatal)         # a refusal costs one row


def test_a_failed_image_tool_call_relays_its_error(app, monkeypatch):
    from lds_api_engines import chatgpt_image
    r = MagicMock(status_code=200, headers={})
    r.text = _sse({'type': 'response.output_item.done',
                   'item': {'type': 'image_generation_call', 'status': 'failed',
                            'error': {'message': 'image was blocked by policy'}}})
    with pytest.raises(chatgpt_image.ChatGPTImageRefused) as e:
        _sub(monkeypatch, return_value=r)
    assert 'image was blocked by policy' in str(e.value)


def test_a_subscription_401_still_refreshes_once_before_giving_up(app, monkeypatch):
    """The refresh is the whole reason a first 401 is not an error. Only the
    SECOND one is the user's problem, and only then does it say so."""
    from lds_api_engines import chatgpt_image, chatgpt_oauth
    calls = []
    monkeypatch.setattr(chatgpt_oauth, 'access_token',
                        lambda force_refresh=False: calls.append(force_refresh) or TOKEN)
    monkeypatch.setattr(chatgpt_oauth, 'account_id', lambda: 'acc-x')
    monkeypatch.setattr(chatgpt_oauth, 'status',
                        lambda: {'connected': True, 'email': None, 'plan': None})
    ok = MagicMock(status_code=200, headers={})
    ok.text = _sub_image_sse()
    with patch('lds_api_engines.chatgpt_image.requests.post',
               side_effect=[_resp(401, text='expired'), ok]):
        assert chatgpt_image.generate_variation(
            b'r', 'p', force_lane='subscription') == PNG
    assert calls == [False, True]

    with patch('lds_api_engines.chatgpt_image.requests.post',
               side_effect=[_resp(401, text='expired'), _resp(401, text='expired')]):
        with pytest.raises(chatgpt_image.SubscriptionUnavailable) as e:
            chatgpt_image.generate_variation(b'r', 'p', force_lane='subscription')
    assert 'after refreshing the token' in str(e.value)


# --- 3. no message anywhere promises a workaround ---------------------------

def _every_chatgpt_message(app, monkeypatch):
    """Every user-facing sentence both lanes can produce, collected by running
    them. Collected, not listed: a message added later is covered on its own."""
    from lds_api_engines import chatgpt_image
    out = []
    sub_cases = [
        {'return_value': _resp(s, text='x')} for s in (400, 401, 403, 404, 410,
                                                       429, 500, 502, 503, 418)
    ] + [
        {'side_effect': requests.ConnectionError('down')},
        {'side_effect': requests.ReadTimeout('slow')},
        {'return_value': _sse_refusal_resp()},
    ]
    for kw in sub_cases:
        try:
            _sub(monkeypatch, **kw)
        except Exception as e:                          # noqa: BLE001
            out.append(str(e))
    monkeypatch.setenv('OPENAI_API_KEY', KEY)
    api_cases = [
        {'return_value': _openai_err('nope', status=401)},
        {'return_value': _openai_err('must be verified', status=403)},
        {'return_value': _openai_err('missing', status=404)},
        {'return_value': _openai_err('slow down', status=429)},
        {'return_value': _openai_err('oops', status=500)},
        {'return_value': _openai_err('Supported values are: gpt-image-2', param='model')},
        {'return_value': _openai_err('Invalid image', code='invalid_image')},
        {'return_value': _openai_err('rejected as a result of our safety system',
                                     code='moderation_blocked')},
        {'side_effect': requests.ConnectionError('down')},
    ]
    for kw in api_cases:
        with patch('lds_api_engines.chatgpt_image.requests.post', **kw):
            try:
                chatgpt_image.generate_variation(b'r', 'p', force_lane='api')
            except Exception as e:                      # noqa: BLE001
                out.append(str(e))
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    try:
        chatgpt_image.generate_variation(b'r', 'p', force_lane='api')
    except Exception as e:                              # noqa: BLE001
        out.append(str(e))
    return out


@pytest.mark.parametrize('banned', ['retry', 'try again', 'rephrase', 'usually works'])
def test_no_chatgpt_message_ever_promises_a_workaround(app, monkeypatch, banned):
    """The lesson from the Gemini pass: replacing a silence with a promise we
    cannot keep is worse than the silence. Whether the same call would pass a
    second time is precisely the thing this engine cannot know."""
    messages = _every_chatgpt_message(app, monkeypatch)
    assert len(messages) >= 20                          # the sweep really ran
    for msg in messages:
        assert banned not in msg.lower(), msg


def test_no_message_or_log_record_ever_carries_a_credential(app, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    for msg in _every_chatgpt_message(app, monkeypatch):
        assert KEY not in msg
        assert TOKEN not in msg
    assert KEY not in caplog.text
    assert TOKEN not in caplog.text


# --- 4. the API-key lane, which already worked, still works ------------------

def test_the_api_lane_success_path_is_untouched(app, monkeypatch):
    """The anti-regression that decides whether this change was worth making."""
    monkeypatch.setenv('OPENAI_API_KEY', KEY)
    from lds_api_engines import chatgpt_image
    ok = _resp(200, {'data': [{'b64_json': base64.b64encode(PNG).decode()}]})
    with patch('lds_api_engines.chatgpt_image.requests.post', return_value=ok) as post:
        out = chatgpt_image.generate_variation([b'a', b'b'], 'a portrait',
                                               aspect_ratio='3:4')
    assert out == PNG
    assert '/images/edits' in post.call_args.args[0]
    sent = post.call_args.kwargs
    assert sent['data']['size'] == '1024x1536'
    assert 'input_fidelity' not in sent['data']         # gpt-image-2 400s on it
    assert len(sent['files']) == 2
    assert sent['headers']['Authorization'] == f'Bearer {KEY}'


@pytest.mark.parametrize('status,body,expected,fatal', [
    (401, _openai_err('Incorrect API key', status=401), 'rejected the API key', True),
    (403, _openai_err('must be verified', status=403), 'organization verification', True),
    (404, _openai_err('no such model', status=404), 'does not serve the model', True),
    (429, _openai_err('slow down', status=429), 'rate-limited', False),
])
def test_the_api_lane_keeps_every_verdict_it_already_had(app, monkeypatch, status,
                                                         body, expected, fatal):
    monkeypatch.setenv('OPENAI_API_KEY', KEY)
    from lds_api_engines import chatgpt_image
    from app.services.engine_errors import EngineFatal
    with patch('lds_api_engines.chatgpt_image.requests.post', return_value=body):
        with pytest.raises(chatgpt_image.ChatGPTImageError) as e:
            chatgpt_image.generate_variation(b'r', 'p', force_lane='api')
    assert expected in str(e.value)
    assert isinstance(e.value, EngineFatal) is fatal


def test_a_model_blaming_400_is_still_fatal_and_still_blames_the_model(app, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', KEY)
    from lds_api_engines import chatgpt_image
    with patch('lds_api_engines.chatgpt_image.requests.post',
               return_value=_openai_err("Supported values are: 'gpt-image-2'",
                                        param='model')):
        with pytest.raises(chatgpt_image.ChatGPTImageFatal) as e:
            chatgpt_image.generate_variation(b'r', 'p', model='dall-e-3',
                                             force_lane='api')
    assert 'dall-e-3' in str(e.value)
    assert 'image input' in str(e.value)


# --- 5. the 400 that used to mean four things -------------------------------

def test_a_moderation_400_is_named_a_refusal_and_does_not_stop_the_batch(app, monkeypatch):
    """It used to return None: the tile said "empty response", which reads as an
    app failure. It is not — it is OpenAI declining, and it must stay per-row."""
    from lds_api_engines import chatgpt_image
    from app.services.engine_errors import EngineFatal, EngineRefused
    monkeypatch.setenv('OPENAI_API_KEY', KEY)
    with patch('lds_api_engines.chatgpt_image.requests.post',
               return_value=_openai_err(
                   'Your request was rejected as a result of our safety system',
                   code='moderation_blocked')):
        with pytest.raises(chatgpt_image.ChatGPTImageRefused) as e:
            chatgpt_image.generate_variation(b'r', 'p', force_lane='api')
    assert 'safety system refused' in str(e.value)
    assert 'not configurable' in str(e.value)
    assert isinstance(e.value, EngineRefused)
    assert not isinstance(e.value, EngineFatal)


@pytest.mark.parametrize('kwargs', [
    {'code': 'invalid_image'},
    {'param': 'image[]'},
    {'message_override': 'Invalid image: the uploaded file could not be decoded'},
])
def test_a_broken_reference_photo_is_not_reported_as_a_refused_prompt(app, monkeypatch,
                                                                     kwargs):
    """The 400 that used to be lumped in with moderation. The remedy is at the
    opposite end of the app — a file to replace, not a prompt to reconsider —
    and since every row of a batch is sent the same references, it repeats."""
    from lds_api_engines import chatgpt_image
    from app.services.engine_errors import EngineFatal, EngineRefused
    monkeypatch.setenv('OPENAI_API_KEY', KEY)
    msg = kwargs.pop('message_override', 'bad input')
    with patch('lds_api_engines.chatgpt_image.requests.post',
               return_value=_openai_err(msg, **kwargs)):
        with pytest.raises(chatgpt_image.ChatGPTImageFatal) as e:
            chatgpt_image.generate_variation(b'r', 'p', force_lane='api')
    assert 'reference photos' in str(e.value)
    assert 'the prompt is not what was refused' in str(e.value)
    assert isinstance(e.value, EngineFatal)
    assert not isinstance(e.value, EngineRefused)


def test_a_400_naming_no_cause_stays_honestly_unexplained(app, monkeypatch):
    """The honest half of the split. When OpenAI says nothing we recognise, the
    engine says nothing it cannot support — it keeps returning None, and the
    fan-out states the ambiguity in words instead of picking a side."""
    from lds_api_engines import chatgpt_image
    monkeypatch.setenv('OPENAI_API_KEY', KEY)
    with patch('lds_api_engines.chatgpt_image.requests.post',
               return_value=_openai_err('something went wrong')):
        assert chatgpt_image.generate_variation(b'r', 'p', force_lane='api') is None
    from app.services.face_dataset_service import _EMPTY_MSG
    assert 'look identical here' in _EMPTY_MSG
    assert 'retry' not in _EMPTY_MSG.lower()


def test_a_200_with_no_image_and_no_words_also_stays_none(app, monkeypatch):
    """Same rule on the other lane: unreadable is reported as unreadable."""
    r = MagicMock(status_code=200, headers={})
    r.text = _sse({'type': 'response.completed', 'response': {'output': []}})
    assert _sub(monkeypatch, return_value=r) is None


@pytest.mark.parametrize('text', [
    '', 'not json at all', 'data: {broken', 'data: null\n', 'data: 42\n',
    'data: {"type":"response.completed","response":null}\n',
    'data: {"type":"response.output_item.done","item":"not-a-dict"}\n',
    'data: {"type":"response.output_item.done","item":{"content":"not-a-list"}}\n',
    '{"output": "not-a-list"}',
])
def test_a_malformed_subscription_body_degrades_to_none_not_a_crash(app, monkeypatch, text):
    """This runs on whatever the undocumented backend sends. A shape we did not
    anticipate must come back as "no image", never as an AttributeError the user
    reads as an app bug."""
    r = MagicMock(status_code=200, headers={})
    r.text = text
    r.json.side_effect = ValueError('no json')
    assert _sub(monkeypatch, return_value=r) is None


# --- 6. the lanes agree, event by event -------------------------------------

@pytest.mark.parametrize('event', ['network', '5xx'])
def test_both_lanes_answer_the_same_event_the_same_way(app, monkeypatch, event):
    """The inconsistency this file was opened for, stated as one assertion: same
    file, same event, same KIND of answer. The wording differs (each names its
    own lane); what must not differ is raising versus staying silent."""
    from lds_api_engines import chatgpt_image
    from app.services.engine_errors import EngineFatal, EngineRefused
    kw = ({'side_effect': requests.ConnectionError('down')} if event == 'network'
          else {'return_value': _openai_err('boom', status=500)})
    sub_kw = ({'side_effect': requests.ConnectionError('down')} if event == 'network'
              else {'return_value': _resp(500, text='boom')})

    with pytest.raises(chatgpt_image.ChatGPTImageError) as sub_e:
        _sub(monkeypatch, **sub_kw)
    monkeypatch.setenv('OPENAI_API_KEY', KEY)
    with patch('lds_api_engines.chatgpt_image.requests.post', **kw):
        with pytest.raises(chatgpt_image.ChatGPTImageError) as api_e:
            chatgpt_image.generate_variation(b'r', 'p', force_lane='api')

    for e in (sub_e.value, api_e.value):
        assert not isinstance(e, EngineRefused)         # neither is a refusal
        assert not isinstance(e, EngineFatal)           # neither stops the run
