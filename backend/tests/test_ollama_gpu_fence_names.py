"""The GPU fence compares model names the way Ollama prints them.

Reported on Discord #help (2026-09-03): a Bank caption pass captioned ONE image
and finished; every image after it was refused as "a local Ollama model already
in use outside LDS", and the ComfyUI queue stayed blocked behind it. Killing
Ollama bought exactly one more image.

The configured model was `qwen3-vl-abliterated`, typed without a tag — the
Settings field is free text and the Test button accepted it. Ollama does not
answer with the name you type: it fills the implicit tag in and elides the
default registry and namespace, so `/api/ps` reports `qwen3-vl-abliterated:latest`
(measured on a real daemon on 2026-09-03: a model requested as `ldsfencetest`
comes back as `ldsfencetest:latest`; `types/model/name.go`, DisplayShortest).
The fence remembered the typed string and compared byte-for-byte, so from the
second image on LDS read its OWN model as a stranger's and fenced itself out of
it — admission refused, and the ComfyUI hand-off refused to release a residency
it did not recognise as its own.

The fence now writes its own model down the way Ollama will print it back. It
does NOT adopt the Test button's "a tagless name matches any tag": a different
tag is a different residency, and adopting it would let LDS unload a model
somebody else loaded — the one direction this module's docstring forbids.
"""
import json
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from app.services import ollama_gpu_fence as fence

# Drives /api/ps through `requests` itself, like the fence's own test file.
pytestmark = pytest.mark.ollama_fence

ENDPOINT = 'http://127.0.0.1:11434'


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


def _ps(*names):
    return _Response({'models': [{'name': name} for name in names]})


def _ps_with_expiry(name, expires_at):
    return _Response({'models': [{'name': name, 'expires_at': expires_at}]})


def _iso(seconds_from_now):
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds_from_now)).isoformat()


@pytest.fixture(autouse=True)
def _reset_fence():
    fence.reset_for_tests()
    yield
    fence.reset_for_tests()


@pytest.fixture
def fence_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(fence, '_claims_path', lambda: tmp_path / 'ollama_fence_claims.json')
    return tmp_path


# --- the name itself ---------------------------------------------------------
@pytest.mark.parametrize('typed, printed', [
    ('llava', 'llava:latest'),
    ('ldsfencetest', 'ldsfencetest:latest'),                 # measured, 2026-09-03
    ('qwen3-vl-abliterated', 'qwen3-vl-abliterated:latest'),  # the reported config
    ('llava:13b', 'llava:13b'),
    ('huihui_ai/qwen3-vl-abliterated', 'huihui_ai/qwen3-vl-abliterated:latest'),
    ('huihui_ai/qwen3-vl-abliterated:8b', 'huihui_ai/qwen3-vl-abliterated:8b'),
    ('library/llava', 'llava:latest'),
    ('registry.ollama.ai/library/llava', 'llava:latest'),
    ('Registry.Ollama.AI/Library/llava:13b', 'llava:13b'),   # Ollama folds these two case-insensitively
    ('registry.ollama.ai/huihui_ai/x', 'huihui_ai/x:latest'),
    ('https://registry.ollama.ai/library/llava:latest', 'llava:latest'),
    ('localhost:5000/org/x', 'localhost:5000/org/x:latest'),  # a custom registry stays named
    ('example.com/ns/x:t', 'example.com/ns/x:t'),
    ('  llava  ', 'llava:latest'),
    ('LLaVA', 'LLaVA:latest'),                               # the model's own case is kept, as Ollama keeps it
    ('llava/', 'llava/'),                                    # no model part: never invent a name
    # A separator that promises a part and delivers none is not a name to Ollama
    # (its parser files the gap as a missing part and refuses the request), so
    # it must not become one here either — see the test below for why.
    ('llava:', 'llava:'),
    ('llava:latest:', 'llava:latest:'),
    ('/llava', '/llava'),
    ('hf.co//r', 'hf.co//r'),
    ('//ZZ', '//ZZ'),
    ('', ''),
])
def test_canonical_name_is_the_name_ollama_prints(typed, printed):
    assert fence.canonical_ollama_name(typed) == printed


def test_a_setting_ollama_would_refuse_never_becomes_a_real_name():
    """`llava:` passes the Settings Test button green (the probe folds it) and
    Ollama refuses it, so nothing is ever loaded under it. Had the fence filled
    the gap with `latest`, LDS would have OWNED `llava:latest` without loading
    it — and the next hand-off would have unloaded another tool's copy."""
    with patch.object(fence.requests, 'get', return_value=_ps()):
        assert fence.mark_before_generate(ENDPOINT, 'llava:') == 'local'
    assert fence._owned_models[ENDPOINT] == {'llava:'}
    with patch.object(fence.requests, 'get', return_value=_ps('llava:latest')), \
            patch.object(fence.requests, 'post') as post, \
            patch.object(fence, '_configured_local_endpoint', return_value=('local', ENDPOINT)):
        assert fence.mark_before_generate(ENDPOINT, 'llava:') == 'blocked'
        assert fence.ensure_released_for_comfy() is False
    post.assert_not_called()


# --- the reported sequence ---------------------------------------------------
def test_lds_recognises_its_own_model_under_the_name_ollama_gives_it_back():
    """An empty runner, LDS loads `qwen3-vl-abliterated`, Ollama reports it as
    `qwen3-vl-abliterated:latest` — and the next image is still LDS's to run."""
    with patch.object(fence.requests, 'get', return_value=_ps()):
        assert fence.mark_before_generate(ENDPOINT, 'qwen3-vl-abliterated') == 'local'
    with patch.object(fence.requests, 'get', return_value=_ps('qwen3-vl-abliterated:latest')), \
            patch.object(fence.requests, 'post') as post:
        assert fence.mark_before_generate(ENDPOINT, 'qwen3-vl-abliterated') == 'local'
    post.assert_not_called()
    assert ENDPOINT not in fence._foreign_local_endpoints


def test_the_comfy_handoff_releases_the_model_lds_loaded_without_a_tag():
    """...and the queue behind it: the hand-off unloads that residency by the
    name Ollama uses and proves the runner empty, instead of refusing it as a
    stranger's and holding every ComfyUI job."""
    with patch.object(fence.requests, 'get', return_value=_ps()):
        assert fence.mark_before_generate(ENDPOINT, 'qwen3-vl-abliterated') == 'local'
    with patch.object(fence.requests, 'get',
                      side_effect=[_ps('qwen3-vl-abliterated:latest'), _ps()]), \
            patch.object(fence.requests, 'post', return_value=_Response({})) as post, \
            patch.object(fence, '_configured_local_endpoint', return_value=('local', ENDPOINT)):
        assert fence.ensure_released_for_comfy() is True
    assert post.call_args.kwargs['json'] == {'model': 'qwen3-vl-abliterated:latest',
                                             'keep_alive': 0}
    assert fence.last_block() is None


def test_a_caption_batch_cleanup_finds_the_model_under_its_typed_name():
    """unload_vision_model asks by the configured (typed) name; the residency it
    has to release is filed under the printed one."""
    with patch.object(fence.requests, 'get', return_value=_ps()):
        fence.mark_before_generate(ENDPOINT, 'qwen3-vl-abliterated')
    with patch.object(fence.requests, 'get',
                      side_effect=[_ps('qwen3-vl-abliterated:latest'), _ps()]), \
            patch.object(fence.requests, 'post', return_value=_Response({})) as post:
        assert fence.release_owned_models(ollama_url=ENDPOINT,
                                          model='qwen3-vl-abliterated') is True
    assert post.call_count == 1
    assert fence._owned_models.get(ENDPOINT, set()) == set()


def test_fence_status_does_not_report_lds_own_tagless_model_as_in_the_way():
    with patch.object(fence.requests, 'get', return_value=_ps()):
        fence.mark_before_generate(ENDPOINT, 'qwen3-vl-abliterated')
    with patch.object(fence, '_configured_local_endpoint', return_value=('local', ENDPOINT)), \
            patch.object(fence.requests, 'get', return_value=_ps('qwen3-vl-abliterated:latest')):
        state = fence.fence_status()
    assert state['blocked'] is False and state['models'] == []


def test_the_registry_and_library_prefixes_fold_the_way_ollama_folds_them():
    typed = 'registry.ollama.ai/library/llava:13b'
    with patch.object(fence.requests, 'get', return_value=_ps()):
        assert fence.mark_before_generate(ENDPOINT, typed) == 'local'
    with patch.object(fence.requests, 'get', return_value=_ps('llava:13b')):
        assert fence.mark_before_generate(ENDPOINT, typed) == 'local'


# --- what must NOT be adopted --------------------------------------------------
@pytest.mark.parametrize('typed, resident', [
    ('llava', 'llava:13b'),                    # an implicit `latest` is not "any tag"
    ('llava:13b', 'llava:latest'),
    ('llava', 'huihui_ai/llava:latest'),       # another publisher's build
    ('localhost:5000/org/x', 'org/x:latest'),  # a custom registry is not the default one
])
def test_a_different_tag_or_publisher_is_still_somebody_elses_model(typed, resident):
    """The Test button's leniency (a tagless name matches any pulled tag) is NOT
    the fence's: adopting `llava:13b` for a typed `llava` would let a ComfyUI
    hand-off unload a model another tool loaded."""
    with patch.object(fence.requests, 'get', return_value=_ps()):
        assert fence.mark_before_generate(ENDPOINT, typed) == 'local'
    with patch.object(fence.requests, 'get', return_value=_ps(resident)), \
            patch.object(fence.requests, 'post') as post, \
            patch.object(fence, '_configured_local_endpoint', return_value=('local', ENDPOINT)):
        assert fence.mark_before_generate(ENDPOINT, typed) == 'blocked'
        assert fence.ensure_released_for_comfy() is False
    post.assert_not_called()
    assert fence.last_block()['models'] == [resident]


def test_lm_studio_identifiers_are_compared_as_typed():
    """Only the Ollama driver prints names back with a tag. An LM Studio id such
    as `qwen/qwen3-vl-8b` is what that server reports, verbatim, and must not
    grow one — or the model LDS borrows would stop matching the one it sees."""
    with patch.object(fence, '_probe_lmstudio',
                      return_value=('models', {'qwen/qwen3-vl-8b'}, {})):
        assert fence.mark_before_generate(ENDPOINT, 'qwen/qwen3-vl-8b',
                                          provider='lmstudio') == 'local'
    assert fence._borrowed_models[ENDPOINT] == {'qwen/qwen3-vl-8b'}
    assert fence._canonical_for(ENDPOINT, 'qwen/qwen3-vl-8b') == 'qwen/qwen3-vl-8b'


def test_lm_studio_load_is_filed_under_the_pinned_driver_as_typed():
    """register_lds_load is LM Studio's by contract; it pins the driver, so the
    identifier is filed exactly as the admission will look it up — the one
    writer of the ownership map that used to bypass the canonical door."""
    fence.register_lds_load(ENDPOINT, 'qwen/qwen3-vl-8b')
    assert fence._driver_for(ENDPOINT) == 'lmstudio'
    assert fence._owned_models[ENDPOINT] == {'qwen/qwen3-vl-8b'}
    with patch.object(fence, '_probe_lmstudio',
                      return_value=('models', {'qwen/qwen3-vl-8b'}, {})):
        assert fence.mark_before_generate(ENDPOINT, 'qwen/qwen3-vl-8b',
                                          provider='lmstudio') == 'local'
    # Its own load, not a borrowed one: the hand-off may release it.
    assert 'qwen/qwen3-vl-8b' not in fence._borrowed_models.get(ENDPOINT, set())


# --- the claims that outlive a restart -------------------------------------------
def test_a_new_claim_is_filed_under_the_printed_name(fence_data_dir):
    """...so the next start reads it against /api/ps without a translation step."""
    with patch.object(fence.requests, 'get', return_value=_ps()):
        fence.mark_before_generate(ENDPOINT, 'qwen3-vl-abliterated', keep_alive='120s')
    assert list(fence._read_claims()[ENDPOINT]) == ['qwen3-vl-abliterated:latest']


def test_a_claim_written_under_the_typed_name_still_speaks_after_a_restart(fence_data_dir):
    """A claim persisted by a build that stored the typed name must not fence the
    upgraded build out of its own warm model for the rest of the lease."""
    fence._claims_path().write_text(json.dumps({'version': 1, 'claims': {
        ENDPOINT: {'qwen3-vl-abliterated': {'deadline': time.time() + 60}}}}),
        encoding='utf-8')
    with patch.object(fence.requests, 'get',
                      return_value=_ps_with_expiry('qwen3-vl-abliterated:latest', _iso(50))), \
            patch.object(fence.requests, 'post') as post:
        assert fence.mark_before_generate(ENDPOINT, 'qwen3-vl-abliterated',
                                          keep_alive='120s') == 'local'
    post.assert_not_called()
