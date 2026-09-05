"""Opt-in check against a REAL local Ollama. Skipped by default, on purpose.

The unit suite is now hermetic: it never reads this machine's Ollama, because
letting it do so made ~24 tests pass or fail on whatever model happened to be
resident. That hermeticity is worth keeping, but it also means nothing in the
suite proves the fence still speaks Ollama's ACTUAL protocol — that /api/ps
answers the shape `_probe` parses, and that a `keep_alive: 0` generate really
frees the runner. A stub can drift from the daemon without either changing.

So the live proof stays available and is never automatic:

    LDS_TEST_LIVE_OLLAMA=1 python -m pytest tests/test_ollama_fence_live.py

It is read-only unless it loaded the model itself, and it refuses to run at all
when a model it did not load is resident — evicting someone else's work is the
exact thing the fence exists to prevent, and a test is not an exception.
"""
import os
from unittest.mock import patch

import pytest

# `ollama_http` as well as `ollama_fence`: the conftest guard that keeps the unit
# suite off this machine's daemon refuses `requests` itself unless a test carries
# it, and without it every test here skipped with "no local Ollama answering"
# on a machine whose Ollama was answering — an opt-in proof that could never run.
pytestmark = [pytest.mark.ollama_fence, pytest.mark.ollama_http, pytest.mark.live_ollama,
              pytest.mark.skipif(os.environ.get('LDS_TEST_LIVE_OLLAMA') != '1',
                                 reason='set LDS_TEST_LIVE_OLLAMA=1 to run against a real Ollama')]

ENDPOINT = 'http://127.0.0.1:11434'


def _ps():
    import requests
    return requests.get(f'{ENDPOINT}/api/ps', timeout=(3, 5))


def test_probe_parses_what_a_real_ollama_actually_answers():
    """The shape contract, against the daemon rather than a fixture."""
    from app.services import ollama_gpu_fence as fence
    try:
        response = _ps()
    except Exception as exc:                                   # pragma: no cover
        pytest.skip(f'no local Ollama answering: {exc}')
    assert response.status_code == 200

    state, names, expiry = fence._probe(ENDPOINT)
    assert state in ('empty', 'models')
    live = {m.get('name') or m.get('model') for m in response.json().get('models', [])}
    assert names == {n for n in live if n}
    # Every residency Ollama reports must be readable as a deadline, or the
    # restart-adoption guard silently loses the fact it is built on.
    for name in names:
        assert expiry.get(name) is not None, f'unparseable expires_at for {name}'


def test_a_model_this_test_did_not_load_is_left_strictly_alone():
    from app.services import ollama_gpu_fence as fence
    try:
        state, names, _ = fence._probe(ENDPOINT)
    except Exception as exc:                                   # pragma: no cover
        pytest.skip(f'no local Ollama answering: {exc}')
    if state != 'models':
        pytest.skip('nothing resident: load a model to exercise the refusal')
    verdict = fence.mark_before_generate(ENDPOINT, 'lds-not-this-one', keep_alive=0)
    assert verdict == 'blocked'
    # And it is still there: the refusal touched nothing.
    assert fence._probe(ENDPOINT)[1] == names


def test_the_daemon_prints_a_model_back_the_way_the_fence_expects():
    """The fact `canonical_ollama_name` is built on, measured on the daemon
    rather than remembered: a model asked for under a spelling Ollama merely
    tolerates (default registry and namespace written out, an implicit tag) is
    reported by /api/ps under its shortest form — and the fence, which files
    its own model under that form, keeps recognising it once it is resident.
    Before the fix the second admission below answered `blocked`, which is how
    a Bank caption pass came to caption one image and refuse the rest.

    Loads the smallest pulled model for real (an empty prompt is a load), only
    on an EMPTY runner, and unloads it itself afterwards."""
    import requests
    from app.services import ollama_gpu_fence as fence
    try:
        state, names, _ = fence._probe(ENDPOINT)
        tags = requests.get(f'{ENDPOINT}/api/tags', timeout=(3, 5)).json().get('models', [])
    except Exception as exc:                                   # pragma: no cover
        pytest.skip(f'no local Ollama answering: {exc}')
    if state != 'empty':
        pytest.skip(f'runner not empty ({sorted(names)}): never evict somebody else\'s model')
    if not tags:
        pytest.skip('no model pulled on this daemon')
    printed = min(tags, key=lambda m: m.get('size', 0))['name']
    # The same model under the spelling the daemon does NOT print: registry and
    # `library/` written out, and the tag dropped when it is the implicit one.
    typed = printed[:-len(':latest')] if printed.endswith(':latest') else printed
    typed = ('registry.ollama.ai/' + ('' if '/' in typed else 'library/') + typed)
    assert fence.canonical_ollama_name(typed) == printed
    assert typed != printed

    fence.reset_for_tests()
    assert fence.mark_before_generate(ENDPOINT, typed, keep_alive='30s') == 'local'
    try:
        load = requests.post(f'{ENDPOINT}/api/generate',
                             json={'model': typed, 'prompt': '', 'keep_alive': '30s'},
                             timeout=(10, 300))
        assert load.status_code == 200, load.text[:200]
        assert fence._probe(ENDPOINT)[1] == {printed}
        # The reported failure: LDS's own residency, read under the printed name.
        assert fence.mark_before_generate(ENDPOINT, typed, keep_alive='30s') == 'local'
        with patch.object(fence, '_configured_local_endpoint', return_value=('local', ENDPOINT)):
            status = fence.fence_status()
        assert status['reachable'] is True and status['blocked'] is False, status
    finally:
        assert fence.release_owned_models(ollama_url=ENDPOINT, model=typed) is True
    assert fence._probe(ENDPOINT)[0] in ('empty', 'down')
