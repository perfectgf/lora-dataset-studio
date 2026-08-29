"""The suite must never open a socket to this machine's Ollama.

Two guards in conftest make that true — one for `ollama_gpu_fence`, one for
`capabilities` and `vision_ollama` — and each was written after the same lesson
arrived twice: a test whose verdict depends on a daemon nobody declared is not a
test, it is a reading of the developer's machine. The fence one (2026-08-02) was
measured at 24 tests flipping on whether a model happened to be resident. The
second (2026-08-29) was found with a socket tripwire during a hygiene pass: 119
real connections over a full run, 59 tests, 24 files.

The cheap half was 118 GET /api/tags. The half that matters was a single POST
/api/generate — an image upload reaching `detect_head_bbox`, which LOADS the
8 GB vision model onto the shared GPU. A unit suite that can occupy the card is
not merely impure: it slows, and can stall, whatever else is training or
generating on the same machine.

What is asserted here is the PROPERTY, not the implementation: no connection to
port 11434 leaves the probes. A future refactor may move the seam; it may not
re-open the socket. Without this file the guards are two fixtures nobody would
notice deleting, and the leak returns in silence — which is exactly how it came
back the second time.
"""
import socket

import pytest

OLLAMA_PORT = 11434


class _Tripwire:
    """Records every connect() to Ollama's port instead of making it.

    Records rather than raises: a raise would be caught by the probe's own
    try/except and reported as 'unreachable', which is what a passing test
    looks like — the failure has to be visible to the ASSERTION, not to the
    code under test.
    """

    def __init__(self):
        self.hits = []
        self._real = socket.socket.connect

    def __enter__(self):
        tripwire = self

        def connect(sock, address):
            if (isinstance(address, tuple) and len(address) >= 2
                    and address[1] == OLLAMA_PORT):
                tripwire.hits.append(address)
                raise ConnectionRefusedError('tripwire: the suite must not reach Ollama')
            return tripwire._real(sock, address)

        socket.socket.connect = connect
        return self

    def __exit__(self, *exc):
        socket.socket.connect = self._real
        return False


def test_the_reachability_probe_never_touches_the_daemon(app):
    """`probe_ollama` reads {url}/api/tags. Under the guard it must answer
    without a socket, and answer the same thing CI sees: unreachable."""
    from app import capabilities as caps
    with app.app_context(), _Tripwire() as wire:
        verdict = caps.probe_ollama()
    assert wire.hits == [], f'the suite opened a socket to Ollama: {wire.hits}'
    assert verdict['ok'] is False


def test_the_model_probe_never_touches_the_daemon(app):
    """`probe_ollama_model` reaches for the tag list to decide whether the
    vision model is pulled — the second seam, and the one that answers a
    question about the developer's disk rather than the code."""
    from app import capabilities as caps
    with app.app_context(), _Tripwire() as wire:
        verdict = caps.probe_ollama_model()
    assert wire.hits == [], f'the suite opened a socket to Ollama: {wire.hits}'
    assert verdict['ok'] is False


def test_a_full_capabilities_payload_stays_hermetic(app):
    """The path that actually leaked: nobody in those tests asked about Ollama,
    they asked for capabilities and it probed on their behalf."""
    from app import capabilities as caps
    with app.app_context(), _Tripwire() as wire:
        caps.probe(force=True)
    assert wire.hits == [], f'building capabilities reached Ollama: {wire.hits}'


def test_the_guard_does_not_blind_the_rest_of_the_suite(app):
    """The guard refuses port 11434 and delegates everything else untouched.

    Written one notch too wide it would silently blind ComfyUI, Hugging Face and
    the cloud provider — every one of which goes through the same `requests`.
    ComfyUI's own reachability probe shares `_http_ok` with Ollama's, which
    makes it the sharpest case to pin."""
    from app import capabilities as caps
    seen = []
    with app.app_context():
        # A URL that is not Ollama's must reach the real helper, whatever it
        # then answers about a server that is not running.
        original = caps.requests.get

        def spy(url, *a, **k):
            seen.append(url)
            raise OSError('no server here, and that is fine')

        caps.requests.get = spy
        try:
            caps._http_ok('http://127.0.0.1:8188/history')
        finally:
            caps.requests.get = original
    assert seen and seen[0].endswith('/history'), (
        'the Ollama guard swallowed a ComfyUI probe')


def test_a_head_crop_never_loads_the_vision_model(app, tmp_path):
    """THE expensive one. `detect_head_bbox` POSTs /api/generate, and every
    image upload goes through it — so two ordinary route tests (an export
    content-type check and a multipart-parsing check, neither of them about
    vision) were loading 8 GB of weights onto the GPU of whoever ran the suite.

    Asserting on the socket rather than on the stub is deliberate: the seam
    moved once already (the caller imports it inside the function), and the
    property that matters is "no traffic", not "this symbol is a lambda"."""
    from PIL import Image
    from app.services import face_dataset_service as fds
    src = tmp_path / 'face.png'
    Image.new('RGB', (64, 64), 'grey').save(src)
    with app.app_context(), _Tripwire() as wire:
        fds.detect_head_bbox(src.read_bytes())
    assert wire.hits == [], (
        f'a head crop reached Ollama and would load the vision model: {wire.hits}')


def test_the_refusal_looks_like_a_daemon_that_is_not_running():
    """The SHAPE of the refusal is load-bearing, not decoration.

    `ollama_gpu_fence._connection_refused` walks the exception chain looking
    for ECONNREFUSED to tell "nothing is listening" from "something broke", and
    files the endpoint as `down` or `unknown` accordingly. A bare
    ConnectionError from this guard would produce `unknown` — a third answer
    that neither a developer machine nor CI ever gives, which is exactly what
    the guard exists to prevent. Pinned against the real consumer rather than
    against the constructor, so the two cannot drift apart."""
    import requests
    from app.services import ollama_gpu_fence as fence
    try:
        requests.get('http://127.0.0.1:11434/api/tags', timeout=1)
    except Exception as exc:
        assert fence._connection_refused(exc), (
            'the guard refuses in a shape the fence reads as `unknown`, not `down`')
    else:
        raise AssertionError('the guard let the request through')


def test_the_guard_is_armed_for_an_ordinary_test():
    """The negative of the test below: without the marker, the guard IS on.
    Asserting only the opt-out would pass just as well on a guard that never
    installed itself."""
    import requests
    assert getattr(requests.get, 'lds_ollama_guard', False), 'the guard is not installed'
    assert getattr(requests.post, 'lds_ollama_guard', False), 'the guard is not installed'


@pytest.mark.ollama_http
def test_a_test_about_these_calls_can_opt_back_in():
    """The escape hatch has to actually escape, or a test that exists to drive
    a real daemon — the live fence check, an integration probe — would be
    asserting against a refused connection instead."""
    import requests
    assert not getattr(requests.get, 'lds_ollama_guard', False), (
        '@pytest.mark.ollama_http did not lift the guard')
    assert not getattr(requests.post, 'lds_ollama_guard', False), (
        '@pytest.mark.ollama_http did not lift the guard')
