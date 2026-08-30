"""LDS loads the LM Studio model itself — the end of "load it by hand, again".

The complaint that produced this, verbatim from the first real install: "why do
I have to keep loading a model? LDS is supposed to handle everything." It was
right twice over — on principle, and on consistency: the driver already starts
the server and unloads models, so refusing to LOAD one was a gap, not caution.

Every hazard asserted here was measured on a live 0.4.23 before the code was
written; the docstrings name which.
"""
from urllib.parse import urlsplit

import pytest

from app import config
from app.services import ollama_gpu_fence as fence
from app.services import vision_lmstudio as lms
from app.services import vision_llm

pytestmark = pytest.mark.ollama_fence   # these drive the real fence on purpose

URL = 'http://127.0.0.1:1299'


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(lms, '_seen_loaded', set())
    fence._owned_models.clear()
    fence._borrowed_models.clear()
    fence._endpoint_driver.clear()
    yield
    fence._owned_models.clear()
    fence._borrowed_models.clear()
    fence._endpoint_driver.clear()


def _server(loaded, *, surface='v1', calls=None):
    """A fake LM Studio: GET lists, POST /load records and flips residency."""
    state = {'loaded': bool(loaded)}

    class _R:
        def __init__(self, code, body):
            self.status_code, self._body, self.text = code, body, str(body)

        def json(self):
            if self._body is None:
                raise ValueError('no json')
            return self._body

    def get(url, *a, **kw):
        path = urlsplit(url).path
        if surface == 'v1' and path == '/api/v1/models':
            inst = [{'id': 'qwen/qwen3-vl-4b'}] if state['loaded'] else []
            return _R(200, {'models': [{'key': 'qwen/qwen3-vl-4b', 'type': 'vlm',
                                        'loaded_instances': inst}]})
        if surface == 'openai' and path == '/v1/models':
            return _R(200, {'object': 'list', 'data': [{'id': 'qwen/qwen3-vl-4b'}]})
        return _R(404, None)

    def post(url, *a, **kw):
        path = urlsplit(url).path
        if calls is not None:
            calls.append((path, (kw.get('json') or {})))
        if path == '/api/v1/models/load':
            name = (kw.get('json') or {}).get('model')
            if name != 'qwen/qwen3-vl-4b':
                return _R(404, {'error': {'type': 'model_not_found',
                                          'message': f'Model {name} not found in downloaded models'}})
            state['loaded'] = True
            return _R(200, {'status': 'loaded', 'instance_id': name,
                            'load_time_seconds': 2.0})
        if path == '/v1/chat/completions':
            if not state['loaded']:
                return _R(400, {'error': 'No models loaded.'})
            return _R(200, {'choices': [{'message': {'content': 'a red square'}}]})
        return _R(404, None)

    return get, post


def _as_lmstudio(app):
    with app.app_context():
        config.save_config({'local_llm': {'provider': 'lmstudio'},
                            'lmstudio': {'url': URL}})


def test_an_absent_model_is_loaded_and_owned(app, monkeypatch):
    """The load happens, and the fence records it as LDS's OWN residency.

    Ownership is not bookkeeping: it is what lets the keep-warm lease unload the
    model later and lets a ComfyUI hand-off actually free the card. Without it
    the auto-loaded model reads as borrowed — the class the fence refuses to
    touch — and the VRAM could never be handed over."""
    calls = []
    get, post = _server(loaded=False, calls=calls)
    monkeypatch.setattr(lms.requests, 'get', get)
    monkeypatch.setattr(lms.requests, 'post', post)
    _as_lmstudio(app)
    with app.app_context():
        ok, detail = lms.ensure_model_loaded('qwen/qwen3-vl-4b', url=URL)
    assert ok and 'loaded' in detail
    assert ('/api/v1/models/load', {'model': 'qwen/qwen3-vl-4b'}) in calls
    assert 'qwen/qwen3-vl-4b' in fence._owned_models.get(URL, set())
    assert 'qwen/qwen3-vl-4b' not in fence._borrowed_models.get(URL, set())


def test_a_loaded_model_is_never_loaded_again(app, monkeypatch):
    """Measured: POST /load on a resident model does not no-op — it stacks a
    second instance (":2") and doubles the VRAM. The residency check is the
    guard, and this test is what keeps it there."""
    calls = []
    get, post = _server(loaded=True, calls=calls)
    monkeypatch.setattr(lms.requests, 'get', get)
    monkeypatch.setattr(lms.requests, 'post', post)
    _as_lmstudio(app)
    with app.app_context():
        ok, detail = lms.ensure_model_loaded('qwen/qwen3-vl-4b', url=URL)
    assert ok and detail == 'already loaded'
    assert not [c for c in calls if c[0] == '/api/v1/models/load']
    # ...and it stays somebody else's residency: the user loaded it, LDS may use
    # it and may never unload it.
    assert 'qwen/qwen3-vl-4b' not in fence._owned_models.get(URL, set())


def test_a_model_that_is_not_downloaded_names_the_one_gesture_left(app, monkeypatch):
    """Downloading stays in LM Studio (it shows progress and lets you cancel).
    The sentence has to say THAT — 'model_not_found' names nothing a user can do."""
    get, post = _server(loaded=False)
    monkeypatch.setattr(lms.requests, 'get', get)
    monkeypatch.setattr(lms.requests, 'post', post)
    _as_lmstudio(app)
    with app.app_context():
        ok, detail = lms.ensure_model_loaded('nope/never-downloaded', url=URL)
    assert not ok
    assert 'not downloaded' in detail and 'LM Studio' in detail


def test_the_openai_only_surface_is_never_blind_loaded(app, monkeypatch):
    """That surface reports no residency, so a blind load risks the double
    instance above — and a JIT-enabled server behind it loads on its own.
    The request is left to the server, exactly as before this feature."""
    calls = []
    get, post = _server(loaded=False, surface='openai', calls=calls)
    monkeypatch.setattr(lms.requests, 'get', get)
    monkeypatch.setattr(lms.requests, 'post', post)
    _as_lmstudio(app)
    with app.app_context():
        ok, detail = lms.ensure_model_loaded('qwen/qwen3-vl-4b', url=URL)
    assert ok and 'OpenAI' in detail
    assert not [c for c in calls if c[0] == '/api/v1/models/load']


def test_describe_heals_the_empty_server_instead_of_lecturing(app, monkeypatch):
    """THE user's loop, closed: server up, nothing loaded, a caption arrives.

    Before: every vision pass failed with "no usable model loaded" and the
    screen told the user to go load one — which the keep-warm lease would then
    unload again. Now the pass loads the model itself and answers."""
    calls = []
    get, post = _server(loaded=False, calls=calls)
    monkeypatch.setattr(lms.requests, 'get', get)
    monkeypatch.setattr(lms.requests, 'post', post)
    monkeypatch.setattr(lms, '_data_uri_ok', True)
    _as_lmstudio(app)
    import io as _io

    from PIL import Image
    buf = _io.BytesIO()
    Image.new('RGB', (64, 64), (200, 30, 30)).save(buf, 'JPEG')
    with app.app_context():
        out = lms.describe_image(buf.getvalue(), 'describe', url=URL, strict=True)
    assert out == 'a red square'
    assert [c for c in calls if c[0] == '/api/v1/models/load'], 'nothing loaded the model'


def test_the_routed_load_reports_when_nothing_is_downloaded(app, client, monkeypatch):
    """The button's empty case: reachable server, empty disk. Always 200 — the
    remedy rides in the body, never in a 5xx that stacks a generic toast."""
    monkeypatch.setattr(vision_llm, 'provider', lambda: 'lmstudio')
    from app.services import vision_lmstudio
    monkeypatch.setattr(vision_lmstudio, 'resolve_model', lambda *a, **k: '')
    r = client.post('/api/local-llm/load')
    assert r.status_code == 200
    body = r.get_json()
    assert body['ok'] is False and 'download' in body['error'].lower()


def test_the_routed_load_loads_and_names_the_model(app, client, monkeypatch):
    monkeypatch.setattr(vision_llm, 'provider', lambda: 'lmstudio')
    from app.services import vision_lmstudio
    monkeypatch.setattr(vision_lmstudio, 'resolve_model', lambda *a, **k: 'qwen/qwen3-vl-4b')
    monkeypatch.setattr(vision_lmstudio, 'ensure_model_loaded',
                        lambda m, **k: (True, 'loaded in 2.0s'))
    body = client.post('/api/local-llm/load').get_json()
    assert body == {'ok': True, 'model': 'qwen/qwen3-vl-4b', 'detail': 'loaded in 2.0s'}


def test_an_lds_loaded_model_is_claimable_and_releasable(app, monkeypatch, tmp_path):
    """The full circle the design promises: LDS loads → LDS owns → the keep-warm
    claim is written at admission → a ComfyUI hand-off may really free the card.
    A user-loaded model keeps the opposite promise (never unloaded); the two
    must not blur, and `register_lds_load` is the line between them."""
    monkeypatch.setattr(fence, '_claims_path', lambda: tmp_path / 'claims.json')
    get, post = _server(loaded=True)
    monkeypatch.setattr(lms.requests, 'get', get)
    monkeypatch.setattr(lms.requests, 'post', post)
    _as_lmstudio(app)
    fence.register_lds_load(URL, 'qwen/qwen3-vl-4b')
    with app.app_context():
        scope = fence.mark_before_generate(URL, 'qwen/qwen3-vl-4b',
                                           provider='lmstudio', keep_alive='120s')
    assert scope == 'local'
    assert 'qwen/qwen3-vl-4b' not in fence._borrowed_models.get(URL, set()), (
        'an LDS-loaded model read as borrowed — the ComfyUI hand-off would refuse '
        'to release a model LDS is fully entitled to release')
    assert fence._read_claims() != {}, 'no keep-warm claim was written for our own load'
