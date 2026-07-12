from unittest.mock import patch, MagicMock
import base64


def test_size_for_aspect_three_sizes_only():
    from app.services.chatgpt_image import size_for_aspect
    assert size_for_aspect('1:1') == '1024x1024'
    assert size_for_aspect('3:4') == '1024x1536'
    assert size_for_aspect('16:9') == '1536x1024'


def test_chatgpt_never_sends_input_fidelity(app, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-x')
    from app.services import chatgpt_image
    resp = MagicMock(status_code=200)
    resp.json.return_value = {'data': [{'b64_json': base64.b64encode(b'png').decode()}]}
    with patch('app.services.chatgpt_image.requests.post', return_value=resp) as post:
        out = chatgpt_image.generate_variation(b'ref', 'a portrait', aspect_ratio='3:4')
    assert out == b'png'
    sent = post.call_args
    payload_keys = set((sent.kwargs.get('data') or {}).keys())
    assert 'input_fidelity' not in payload_keys
    assert sent.kwargs['data']['size'] == '1024x1536'
    assert '/images/edits' in sent.args[0]


def test_chatgpt_returns_none_without_key(app, monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    from app.services import chatgpt_image
    with patch('app.services.chatgpt_image.requests.post') as post:
        assert chatgpt_image.generate_variation(b'r', 'p') is None
    post.assert_not_called()


def test_nanobanana_sends_aspect_config(app, monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'g-x')
    from app.services import nanobanana
    inline = {'inlineData': {'data': base64.b64encode(b'img').decode(), 'mimeType': 'image/webp'}}
    resp = MagicMock(status_code=200)
    resp.json.return_value = {'candidates': [{'content': {'parts': [inline]}}]}
    with patch('app.services.nanobanana.requests.post', return_value=resp) as post:
        out = nanobanana.generate_variation([b'a', b'b'], 'p', aspect_ratio='3:4')
    assert out == b'img'
    body = post.call_args.kwargs['json']
    assert body['generationConfig']['imageConfig']['aspectRatio'] == '3:4'
    assert len(body['contents'][0]['parts']) >= 3   # prompt + 2 refs


def test_nanobanana_returns_none_without_key(app, monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    from app.services import nanobanana
    with patch('app.services.nanobanana.requests.post') as post:
        assert nanobanana.generate_variation(b'r', 'p') is None
    post.assert_not_called()


def _sub_connected(monkeypatch):
    """Wire a fake connected subscription into chatgpt_image's oauth module."""
    from app.services import chatgpt_oauth
    monkeypatch.setattr(chatgpt_oauth, 'access_token', lambda force_refresh=False: 'at-x')
    monkeypatch.setattr(chatgpt_oauth, 'account_id', lambda: 'acc-x')
    monkeypatch.setattr(chatgpt_oauth, 'status',
                        lambda: {'connected': True, 'email': 'u@x.io', 'plan': 'plus'})


def _codex_ok_response():
    resp = MagicMock(status_code=200, headers={'content-type': 'application/json'})
    resp.json.return_value = {'output': [
        {'type': 'reasoning'},
        {'type': 'image_generation_call', 'result': base64.b64encode(b'img').decode()},
    ]}
    return resp


def test_chatgpt_auto_routes_to_subscription_when_connected(app, monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    _sub_connected(monkeypatch)
    from app.services import chatgpt_image
    with patch('app.services.chatgpt_image.requests.post',
               return_value=_codex_ok_response()) as post:
        out = chatgpt_image.generate_variation([b'a'] * 7, 'a portrait', aspect_ratio='3:4')
    assert out == b'img'
    url = post.call_args.args[0]
    assert 'chatgpt.com/backend-api/codex/responses' in url
    headers = post.call_args.kwargs['headers']
    assert headers['Authorization'] == 'Bearer at-x'
    assert headers['chatgpt-account-id'] == 'acc-x'
    assert headers['OpenAI-Beta'] == 'responses=experimental'
    assert headers['originator'] == 'codex_cli_rs'
    body = post.call_args.kwargs['json']
    images = [c for c in body['input'][0]['content'] if c['type'] == 'input_image']
    assert len(images) == 5                       # 7 refs capped at SUBSCRIPTION_MAX_REFS
    assert body['input'][0]['content'][-1]['type'] == 'input_text'
    tool = body['tools'][0]
    assert tool['type'] == 'image_generation'
    assert tool['size'] == '1024x1536'
    assert body['tool_choice'] == 'required'
    assert body['store'] is False                 # Codex backend 400s without it


def test_chatgpt_auto_uses_api_key_when_not_connected(app, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-x')
    from app.services import chatgpt_image
    resp = MagicMock(status_code=200)
    resp.json.return_value = {'data': [{'b64_json': base64.b64encode(b'png').decode()}]}
    with patch('app.services.chatgpt_image.requests.post', return_value=resp) as post:
        assert chatgpt_image.generate_variation(b'ref', 'p') == b'png'
    assert '/images/edits' in post.call_args.args[0]


def test_chatgpt_forced_api_ignores_subscription(app, monkeypatch):
    import app.config as cfg
    cfg.save_config({'engines': {'chatgpt_auth': 'api'}})
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-x')
    _sub_connected(monkeypatch)
    from app.services import chatgpt_image
    resp = MagicMock(status_code=200)
    resp.json.return_value = {'data': [{'b64_json': base64.b64encode(b'png').decode()}]}
    with patch('app.services.chatgpt_image.requests.post', return_value=resp) as post:
        assert chatgpt_image.generate_variation(b'ref', 'p') == b'png'
    assert '/images/edits' in post.call_args.args[0]


def test_subscription_429_raises_quota_exceeded(app, monkeypatch):
    import pytest
    _sub_connected(monkeypatch)
    from app.services import chatgpt_image
    resp = MagicMock(status_code=429, text='usage limit reached',
                     headers={'content-type': 'application/json'})
    with patch('app.services.chatgpt_image.requests.post', return_value=resp):
        with pytest.raises(chatgpt_image.SubscriptionQuotaExceeded):
            chatgpt_image.generate_variation(b'r', 'p')


def test_subscription_401_refreshes_and_retries_once(app, monkeypatch):
    from app.services import chatgpt_oauth
    calls = []
    monkeypatch.setattr(chatgpt_oauth, 'access_token',
                        lambda force_refresh=False: calls.append(force_refresh) or 'at-x')
    monkeypatch.setattr(chatgpt_oauth, 'account_id', lambda: 'acc-x')
    monkeypatch.setattr(chatgpt_oauth, 'status',
                        lambda: {'connected': True, 'email': None, 'plan': None})
    from app.services import chatgpt_image
    stale = MagicMock(status_code=401, text='expired', headers={'content-type': ''})
    with patch('app.services.chatgpt_image.requests.post',
               side_effect=[stale, _codex_ok_response()]) as post:
        assert chatgpt_image.generate_variation(b'r', 'p') == b'img'
    assert post.call_count == 2
    assert calls == [False, True]                 # second attempt forces a refresh


def test_subscription_parses_sse_fallback(app, monkeypatch):
    _sub_connected(monkeypatch)
    from app.services import chatgpt_image
    b64 = base64.b64encode(b'sse-img').decode()
    resp = MagicMock(status_code=200, headers={'content-type': 'text/event-stream'})
    resp.text = ('data: {"type":"response.output_item.done","item":'
                 '{"type":"image_generation_call","result":"' + b64 + '"}}\n\n'
                 'data: [DONE]\n')
    with patch('app.services.chatgpt_image.requests.post', return_value=resp):
        assert chatgpt_image.generate_variation(b'r', 'p') == b'sse-img'


def test_subscription_mode_without_connection_returns_none(app, monkeypatch):
    import pytest
    import app.config as cfg
    cfg.save_config({'engines': {'chatgpt_auth': 'subscription'}})
    from app.services import chatgpt_oauth, chatgpt_image
    monkeypatch.setattr(chatgpt_oauth, 'access_token', lambda force_refresh=False: None)
    with patch('app.services.chatgpt_image.requests.post') as post:
        with pytest.raises(chatgpt_image.SubscriptionUnavailable):
            chatgpt_image.generate_variation(b'r', 'p')
    post.assert_not_called()


def test_force_lane_api_ignores_connected_subscription(app, monkeypatch):
    """force_lane='api' pins the API-key lane even when a subscription is
    connected (batch callers pin once so a mid-batch disconnect can't reroute
    later rows)."""
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-x')
    _sub_connected(monkeypatch)
    from app.services import chatgpt_image
    resp = MagicMock(status_code=200)
    resp.json.return_value = {'data': [{'b64_json': base64.b64encode(b'png').decode()}]}
    with patch('app.services.chatgpt_image.requests.post', return_value=resp) as post:
        out = chatgpt_image.generate_variation(b'ref', 'p', force_lane='api')
    assert out == b'png'
    assert '/images/edits' in post.call_args.args[0]


def test_force_lane_subscription_used_even_without_auto_detect(app, monkeypatch):
    """force_lane='subscription' pins the subscription lane and hits the codex
    endpoint when the connection is present."""
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    _sub_connected(monkeypatch)
    from app.services import chatgpt_image
    with patch('app.services.chatgpt_image.requests.post',
               return_value=_codex_ok_response()) as post:
        out = chatgpt_image.generate_variation(b'ref', 'p', force_lane='subscription')
    assert out == b'img'
    assert 'chatgpt.com/backend-api/codex/responses' in post.call_args.args[0]
