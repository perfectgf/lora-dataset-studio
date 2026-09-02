"""The Ollama text driver's payload, seen from the wire.

The motion writer pins that its sampling reaches `vision_llm.generate_text`;
this file pins where each key LANDS. `think` is a top-level key of Ollama's
`/api/generate`, not an option — put inside `options` it is ignored without a
word, and a hybrid model then spends the token budget reasoning about the
format and truncates the prompt. And the captioners' payload, measured as it
is, must not grow a key they never asked for.
"""
from unittest.mock import patch


def _post_capture(app, monkeypatch, **kw):
    from app.services import vision_ollama as vo
    seen = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {'response': 'ok'}

    monkeypatch.setattr(vo, '_admit_local_ollama', lambda *a, **k: None)
    with app.app_context(), \
         patch('app.services.vision_ollama.requests.post',
               side_effect=lambda url, json=None, timeout=None:
               seen.update(url=url, payload=json) or _Resp()):
        out = vo.generate_text_ollama('write it', model='m', **kw)
    assert out == 'ok'
    return seen


def test_the_writer_s_sampling_lands_where_ollama_reads_it(app, monkeypatch):
    seen = _post_capture(app, monkeypatch, top_p=0.8, top_k=20, min_p=0.0,
                         presence_penalty=1.0, think=False, stop=['```'])
    assert seen['url'].endswith('/api/generate')
    p = seen['payload']
    assert p['think'] is False and 'think' not in p['options']
    assert p['options']['top_p'] == 0.8
    assert p['options']['top_k'] == 20
    assert p['options']['min_p'] == 0.0
    assert p['options']['presence_penalty'] == 1.0
    assert p['options']['stop'] == ['```']


def test_the_captioners_payload_gains_nothing_it_did_not_ask_for(app, monkeypatch):
    p = _post_capture(app, monkeypatch)['payload']
    assert 'think' not in p
    for key in ('top_p', 'top_k', 'min_p', 'presence_penalty', 'stop'):
        assert key not in p['options'], key
