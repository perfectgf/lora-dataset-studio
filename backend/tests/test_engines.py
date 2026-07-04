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
