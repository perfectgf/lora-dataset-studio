"""The LM Studio driver, held to what a LIVE 0.4.23 actually does.

Every expectation here was measured on a real install before it was written down
(see the module docstring of vision_lmstudio). Three of them contradict the
public sources, so they are the ones worth pinning:

* the STANDARD data: URI is what works — the open bug report #1752 says the
  opposite, and a driver written from it fails every first call;
* residency is `loaded_instances` on /api/v1 and `state` on /api/v0, two shapes,
  and the OpenAI surface has neither;
* the model type is spelled `embedding` on one surface and `embeddings` on the
  other.

The fail-closed test (`probe_resident` answering `unknown`, never `empty`, when
the server cannot report residency) is the one with teeth: getting it wrong hands
ComfyUI a GPU another process is holding.
"""
import json

import pytest

from app import config
from app.services import vision_lmstudio as lms


class _Resp:
    def __init__(self, status=200, payload=None, text=''):
        self.status_code = status
        self._payload = payload
        self.text = text if text else (json.dumps(payload) if payload is not None else '')

    def json(self):
        if self._payload is None:
            raise ValueError('no json')
        return self._payload


V1_BODY = {'models': [
    {'key': 'qwen/qwen3-vl-4b', 'type': 'vlm', 'display_name': 'Qwen3 VL 4B',
     'loaded_instances': [{'id': 'qwen/qwen3-vl-4b', 'config': {}}]},
    {'key': 'nomic-embed', 'type': 'embedding', 'loaded_instances': []},
]}
V0_BODY = {'object': 'list', 'data': [
    {'id': 'qwen/qwen3-vl-4b', 'type': 'vlm', 'state': 'loaded'},
    {'id': 'nomic-embed', 'type': 'embeddings', 'state': 'not-loaded'},
]}
OPENAI_BODY = {'object': 'list', 'data': [{'id': 'qwen/qwen3-vl-4b', 'object': 'model'}]}


def _router(mapping, missing_status=404):
    """A GET double that answers only the surfaces `mapping` names.

    Matches the PATH, not a suffix. `endswith` made a double mounted on
    `/v1/models` answer `/api/v1/models` too, because the first ends the second —
    so a test that meant to expose only the OpenAI surface silently exposed the
    native one, and the preference order it claimed to prove was never exercised.
    """
    from urllib.parse import urlsplit

    def _get(url, *a, **kw):
        path = urlsplit(url).path
        if path in mapping:
            return _Resp(200, mapping[path])
        return _Resp(missing_status, None, 'not found')
    return _get


# --- URL normalisation: the string LM Studio itself shows the user ------------

@pytest.mark.parametrize('typed, expected', [
    ('http://127.0.0.1:1234', 'http://127.0.0.1:1234'),
    ('http://localhost:1234/v1', 'http://localhost:1234'),      # what the app displays
    ('http://localhost:1234/v1/', 'http://localhost:1234'),
    ('http://localhost:1234/api/v1', 'http://localhost:1234'),
    ('http://localhost:1234/api/v0', 'http://localhost:1234'),
])
def test_the_url_lm_studio_displays_is_accepted_as_typed(typed, expected):
    """LM Studio's Developer tab advertises `.../v1`, so that is what gets pasted.

    Left alone it breaks twice: the driver composes `/v1/v1/chat/completions`, and
    the GPU fence classifies any URL with a path as `unknown` and refuses every
    call with a message about Ollama. One cause, two unrelated-looking symptoms.
    """
    assert lms._suffix_free(typed) == expected


# --- discovery across three surfaces -----------------------------------------

def test_v1_is_preferred_and_reports_residency_from_loaded_instances(app, monkeypatch):
    """Mounts BOTH native surfaces on purpose: with only v1 mounted, "v1 is
    preferred" is not a claim the test can make — v0 was never there to lose."""
    monkeypatch.setattr(lms.requests, 'get',
                        _router({'/api/v1/models': V1_BODY, '/api/v0/models': V0_BODY}))
    with app.app_context():
        out = lms.list_models()
    assert out['surface'] == 'v1'
    vlm = next(m for m in out['models'] if m['id'] == 'qwen/qwen3-vl-4b')
    assert vlm['loaded'] is True
    assert vlm['instances'] == ['qwen/qwen3-vl-4b']   # from `.id`, not `.instance_id`


def test_v0_is_the_fallback_and_reports_residency_from_state(app, monkeypatch):
    monkeypatch.setattr(lms.requests, 'get', _router({'/api/v0/models': V0_BODY}))
    with app.app_context():
        out = lms.list_models()
    assert out['surface'] == 'v0'
    assert next(m for m in out['models'] if m['id'] == 'qwen/qwen3-vl-4b')['loaded'] is True


def test_the_two_spellings_of_the_embedding_type_are_one_type(app, monkeypatch):
    """`embedding` on v1, `embeddings` on v0 — a filter written for one silently
    lets the other through, and an embedding model in a captioning picker is a
    dead choice the user cannot diagnose."""
    monkeypatch.setattr(lms.requests, 'get', _router({'/api/v1/models': V1_BODY}))
    with app.app_context():
        v1 = lms.list_models()
    monkeypatch.setattr(lms.requests, 'get', _router({'/api/v0/models': V0_BODY}))
    with app.app_context():
        v0 = lms.list_models()
    types = lambda o: {m['id']: m['type'] for m in o['models']}   # noqa: E731
    assert types(v1)['nomic-embed'] == types(v0)['nomic-embed'] == 'embeddings'


# --- the fence contract: four states, and `unknown` never means `empty` -------

def test_resident_models_are_reported_with_their_instance_ids(app, monkeypatch):
    monkeypatch.setattr(lms.requests, 'get', _router({'/api/v1/models': V1_BODY}))
    with app.app_context():
        state, names, _ = lms.probe_resident()
    assert state == 'models'
    assert names == ['qwen/qwen3-vl-4b']


def test_an_answering_server_with_nothing_loaded_is_empty(app, monkeypatch):
    empty = {'models': [{'key': 'x', 'type': 'llm', 'loaded_instances': []}]}
    monkeypatch.setattr(lms.requests, 'get', _router({'/api/v1/models': empty}))
    with app.app_context():
        state, names, _ = lms.probe_resident()
    assert (state, names) == ('empty', [])


def test_a_server_that_cannot_report_residency_is_unknown_not_empty(app, monkeypatch):
    """The whole fence is fail-closed. A server answering only the OpenAI surface
    tells us nothing about what holds the card; reading that as 'free' would hand
    ComfyUI a GPU somebody else is using — the one inversion this module exists
    to prevent."""
    monkeypatch.setattr(lms.requests, 'get', _router({'/v1/models': OPENAI_BODY}))
    with app.app_context():
        state, names, meta = lms.probe_resident()
    assert state == 'unknown'
    assert names == []
    assert 'residency' in meta['reason']


def test_only_a_refused_connection_is_down(app, monkeypatch):
    """`down` means "the GPU is free" to the fence — it drops ownership and admits
    ComfyUI on it. Only an actively REFUSED connection proves that, which is the
    shape a machine with nothing listening actually produces (and the shape the
    suite's own guard fakes)."""
    import errno

    def _refused(*a, **kw):
        raise lms.requests.exceptions.ConnectionError(
            ConnectionRefusedError(errno.ECONNREFUSED, 'nothing listening'))

    monkeypatch.setattr(lms.requests, 'get', _refused)
    with app.app_context():
        state, _, _ = lms.probe_resident()
    assert state == 'down'


@pytest.mark.parametrize('failure', ['timeout', 'unauthorized', 'server-error', 'garbage'])
def test_anything_short_of_a_refusal_keeps_the_fence_shut(app, monkeypatch, failure):
    """The inversion this guards against: a 401 from a mistyped token, a read
    timeout, a 500 or a proxy's HTML page all leave LM Studio possibly holding
    gigabytes of VRAM. Filed as `down` they would read as a free card and ComfyUI
    would be admitted onto it — so every one of them must stay `unknown`."""
    def _fail(*a, **kw):
        if failure == 'timeout':
            raise lms.requests.exceptions.ReadTimeout('too slow')
        if failure == 'garbage':
            return _Resp(200, None, '<html>proxy</html>')
        return _Resp(401 if failure == 'unauthorized' else 500, None, 'nope')

    monkeypatch.setattr(lms.requests, 'get', _fail)
    with app.app_context():
        state, names, _ = lms.probe_resident()
    assert state == 'unknown', 'a fail-closed guard must not read "cannot tell" as "free"'
    assert names == []


def test_list_models_never_raises_even_when_nothing_listens(app, monkeypatch):
    """Its docstring promised this and the request was the one part left unguarded,
    so the single most likely failure — LM Studio not started, which this app
    cannot start for the user — escaped as a raw ConnectionError past every
    failure sentence, out to a 500 on the Test button and on the model route."""
    def _boom(*a, **kw):
        raise lms.requests.exceptions.ConnectionError('nothing there')

    monkeypatch.setattr(lms.requests, 'get', _boom)
    monkeypatch.setattr(lms.requests, 'post', _boom)
    with app.app_context():
        listed = lms.list_models()
        assert listed['reachable'] is False and listed['models'] == []
        assert lms.describe_image(_jpeg(), 'describe') == ''      # best-effort
        with pytest.raises(RuntimeError, match='Start Server'):   # strict says the gesture
            lms.describe_image(_jpeg(), 'describe', strict=True)
        assert lms.generate_text('hi') == ''


def test_an_embeddings_model_is_never_elected_as_the_captioner(app, monkeypatch):
    """Keeping an embedding model resident is routine in LM Studio, and
    `vision_model` is empty by default — so "the loaded model" is the default
    path, and picking the embedding one turns every caption into an opaque
    server error with nothing connecting it to the cause."""
    body = {'models': [
        {'key': 'nomic-embed', 'type': 'embedding', 'loaded_instances': [{'id': 'nomic-embed'}]},
        {'key': 'qwen/qwen3-vl-4b', 'type': 'vlm', 'loaded_instances': [{'id': 'qwen/qwen3-vl-4b'}]},
    ]}
    monkeypatch.setattr(lms.requests, 'get', _router({'/api/v1/models': body}))
    with app.app_context():
        assert lms.resolve_model() == 'qwen/qwen3-vl-4b'
    # and with ONLY the embedding model resident, nothing usable is elected
    only_embed = {'models': [body['models'][0]]}
    monkeypatch.setattr(lms.requests, 'get', _router({'/api/v1/models': only_embed}))
    with app.app_context():
        assert lms.resolve_model() == ''


# --- images: the measured direction, and one retry for the builds that differ --

def _sent_image_fields(calls):
    out = []
    for kw in calls:
        for part in kw['json']['messages'][0]['content']:
            if part.get('type') == 'image_url':
                out.append(part['image_url']['url'])
    return out


def test_the_image_goes_out_as_a_standard_data_uri(app, monkeypatch, tmp_path):
    """Measured on 0.4.23: the data URI answers 200 and bare base64 is refused.
    The open bug report #1752 claims the reverse; it is stale for this build, and
    a driver that believed it would fail every first call."""
    calls = []

    def _post(url, *a, **kw):
        calls.append(kw)
        return _Resp(200, {'choices': [{'message': {'content': 'a red square'}}]})

    monkeypatch.setattr(lms.requests, 'get', _router({'/api/v1/models': V1_BODY}))
    monkeypatch.setattr(lms.requests, 'post', _post)
    monkeypatch.setattr(lms, '_data_uri_ok', None)
    with app.app_context():
        out = lms.describe_image(_jpeg(), 'describe')
    assert out == 'a red square'
    assert len(calls) == 1, 'the first shape tried must be the one that works'
    assert _sent_image_fields(calls)[0].startswith('data:image/jpeg;base64,')


def test_a_build_that_refuses_the_data_uri_gets_exactly_one_retry(app, monkeypatch):
    """Belt and braces for the builds where #1752 was live. The retry fires ONLY
    on that error — any other failure must not burn a second inference."""
    calls = []

    def _post(url, *a, **kw):
        calls.append(kw)
        if _sent_image_fields([kw])[0].startswith('data:'):
            return _Resp(400, None, '{"error": "Invalid url."}')
        return _Resp(200, {'choices': [{'message': {'content': 'ok'}}]})

    monkeypatch.setattr(lms.requests, 'get', _router({'/api/v1/models': V1_BODY}))
    monkeypatch.setattr(lms.requests, 'post', _post)
    monkeypatch.setattr(lms, '_data_uri_ok', None)
    with app.app_context():
        assert lms.describe_image(_jpeg(), 'describe') == 'ok'
    assert len(calls) == 2
    fields = _sent_image_fields(calls)
    assert fields[0].startswith('data:') and not fields[1].startswith('data:')


def test_a_failure_that_is_not_about_the_url_does_not_retry(app, monkeypatch):
    calls = []

    def _post(url, *a, **kw):
        calls.append(kw)
        return _Resp(500, None, 'internal error')

    monkeypatch.setattr(lms.requests, 'get', _router({'/api/v1/models': V1_BODY}))
    monkeypatch.setattr(lms.requests, 'post', _post)
    monkeypatch.setattr(lms, '_data_uri_ok', None)
    with app.app_context():
        assert lms.describe_image(_jpeg(), 'describe') == ''
    assert len(calls) == 1


# --- failures say what to do -------------------------------------------------

def test_each_failure_names_the_gesture_that_fixes_it(app):
    with app.app_context():
        assert 'Start Server' in lms.failure_sentence(None, 'timeout')
        assert 'no model loaded' in lms.failure_sentence(400, 'No models loaded. Please load a model')
        assert 'no /v1 suffix' in lms.failure_sentence(404, 'not found')


def test_a_reachable_server_with_nothing_loaded_says_so_when_strict(app, monkeypatch):
    """JIT is off by default, so this is the NORMAL state right after installing
    LM Studio — the most common failure deserves the most precise sentence."""
    empty = {'models': []}
    monkeypatch.setattr(lms.requests, 'get', _router({'/api/v1/models': empty}))
    with app.app_context():
        with pytest.raises(RuntimeError, match='no model loaded'):
            lms.describe_image(_jpeg(), 'describe', strict=True)


# --- unload: the capability Ollama does not have -----------------------------

def test_unload_targets_the_instance_id_and_reports_the_result(app, monkeypatch):
    posted = []

    def _post(url, *a, **kw):
        posted.append((url, kw.get('json')))
        return _Resp(200, {'instance_id': 'qwen/qwen3-vl-4b'})

    monkeypatch.setattr(lms.requests, 'get', _router({'/api/v1/models': V1_BODY}))
    monkeypatch.setattr(lms.requests, 'post', _post)
    with app.app_context():
        assert lms.release() is True
    assert posted[0][0].endswith('/api/v1/models/unload')
    assert posted[0][1] == {'instance_id': 'qwen/qwen3-vl-4b'}


def test_an_already_empty_server_needs_no_unload(app, monkeypatch):
    monkeypatch.setattr(lms.requests, 'get',
                        _router({'/api/v1/models': {'models': []}}))
    with app.app_context():
        assert lms.release() is True


# --- the image gate is not optional for a second provider --------------------

def test_an_unreadable_image_is_never_sent(app, monkeypatch):
    """The gate that strips EXIF/GPS and bounds the payload is shared on purpose:
    a provider that skipped it would turn captioning back into a metadata leak."""
    posted = []
    monkeypatch.setattr(lms.requests, 'get', _router({'/api/v1/models': V1_BODY}))
    monkeypatch.setattr(lms.requests, 'post',
                        lambda *a, **kw: posted.append(kw) or _Resp(200, {}))
    with app.app_context():
        assert lms.describe_image(b'not an image', 'describe') == ''
    assert posted == []


def _jpeg() -> bytes:
    import io as _io

    from PIL import Image
    buf = _io.BytesIO()
    Image.new('RGB', (64, 64), (200, 30, 30)).save(buf, 'JPEG')
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _point_at_a_fixture_server(app):
    """Never the real daemon: the suite's own guard refuses port 1234 anyway."""
    with app.app_context():
        config.save_config({'lmstudio': {'url': 'http://127.0.0.1:1299'}})
    yield
