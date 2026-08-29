from unittest.mock import patch

import pytest

from app.services import ollama_gpu_fence as fence

# This file IS the fence's own test: it drives /api/ps through `requests`
# itself, so it opts out of the autouse stub that keeps the rest of the
# suite off this machine's Ollama (see conftest).
pytestmark = pytest.mark.ollama_fence


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


def _ps(*names):
    return _Response({'models': [{'name': name} for name in names]})


@pytest.fixture(autouse=True)
def _reset_fence():
    fence.reset_for_tests()
    yield
    fence.reset_for_tests()


def test_endpoint_scope_never_treats_a_lookalike_hostname_as_local():
    assert fence._endpoint_scope('http://127.0.0.1:11434')[0] == 'local'
    assert fence._endpoint_scope('http://[::1]:11434') == ('local', 'http://[::1]:11434')
    assert fence._endpoint_scope('http://127.evil.example:11434')[0] == 'remote'
    assert fence._endpoint_scope('http://127.0.0.1:11434/ollama')[0] == 'unknown'
    assert fence._endpoint_scope('http://127.0.0.1:11434/?tenant=lds')[0] == 'unknown'
    assert fence._endpoint_scope('http://localhost:not-a-port')[0] == 'unknown'


def test_transient_probe_failure_can_recover_on_the_next_comfy_handoff():
    endpoint = 'http://127.0.0.1:11434'
    with patch.object(fence.requests, 'get', side_effect=[_Response({}, status_code=500), _ps()]), \
            patch.object(fence, '_configured_local_endpoint', return_value=('local', endpoint)):
        assert fence.mark_before_generate(endpoint, 'lds-model') == 'blocked'
        assert fence.ensure_released_for_comfy() is True

def test_mark_before_generate_refuses_a_preexisting_local_model_without_unload():
    endpoint = 'http://127.0.0.1:11434'
    with patch.object(fence.requests, 'get', return_value=_ps('manual-model')) as get, \
            patch.object(fence.requests, 'post') as post:
        assert fence.mark_before_generate(endpoint, 'lds-model') == 'blocked'
    assert get.call_count == 1
    post.assert_not_called()


def test_comfy_handoff_rechecks_between_cells_and_preserves_manual_model():
    endpoint = 'http://127.0.0.1:11434'
    with patch.object(fence.requests, 'get', side_effect=[_ps(), _ps('manual-model')]) as get, \
            patch.object(fence.requests, 'post') as post:
        assert fence.mark_before_generate(endpoint, 'lds-model') == 'local'
        assert fence.ensure_released_for_comfy() is False
    assert get.call_count == 2
    post.assert_not_called()


def test_comfy_handoff_releases_only_the_model_lds_admitted_then_confirms_empty():
    endpoint = 'http://127.0.0.1:11434'
    with patch.object(fence.requests, 'get', side_effect=[_ps(), _ps('lds-model'), _ps()]), \
            patch.object(fence.requests, 'post', return_value=_Response({})) as post:
        assert fence.mark_before_generate(endpoint, 'lds-model') == 'local'
        assert fence.ensure_released_for_comfy() is True
    post.assert_called_once_with(
        f'{endpoint}/api/generate',
        json={'model': 'lds-model', 'keep_alive': 0},
        timeout=(10, 30), allow_redirects=False)


def test_a_remote_endpoint_is_never_probed_or_unloaded():
    remote = 'http://192.168.10.20:11434'
    with patch.object(fence.requests, 'get') as get, patch.object(fence.requests, 'post') as post:
        assert fence.mark_before_generate(remote, 'remote-model') == 'remote'
        assert fence.release_owned_models(ollama_url=remote) is None
    get.assert_not_called()
    post.assert_not_called()


def test_empty_probe_clears_a_foreign_marker_only_after_it_observes_empty():
    endpoint = 'http://127.0.0.1:11434'
    with fence._lock:
        fence._foreign_local_endpoints.add(endpoint)
    with patch.object(fence.requests, 'get', return_value=_ps()) as get, \
            patch.object(fence.requests, 'post') as post:
        assert fence.release_owned_models(ollama_url=endpoint) is True
    assert get.call_count == 1
    post.assert_not_called()
    with fence._lock:
        assert endpoint not in fence._foreign_local_endpoints


def test_malformed_or_non_successful_probe_blocks_comfy_handoff():
    endpoint = 'http://127.0.0.1:11434'
    with patch.object(fence.requests, 'get', return_value=_Response({}, status_code=500)):
        assert fence.mark_before_generate(endpoint, 'lds-model') == 'blocked'
        assert fence.ensure_released_for_comfy() is False


# --- Ownership that survives a restart ---------------------------------------
# The fence used to keep ownership in process memory only. Restarting LDS while
# its own keep-warm model was still resident therefore made LDS read that model
# as a stranger's and refuse to use it - the app fencing itself out of its own
# GPU allocation, with a message telling the user to unload a model that was
# already theirs. These cover the claim file that closes that hole, and every
# case where the claim must NOT be believed.

def _ps_with_expiry(name, expires_at):
    return _Response({'models': [{'name': name, 'expires_at': expires_at}]})


def _iso(seconds_from_now):
    from datetime import datetime, timedelta, timezone
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds_from_now)).isoformat()


@pytest.fixture
def fence_data_dir(tmp_path, monkeypatch):
    """Point the claim file at a throwaway data dir (never the real one)."""
    monkeypatch.setattr(fence, '_claims_path', lambda: tmp_path / 'ollama_fence_claims.json')
    return tmp_path


def test_keep_alive_seconds_only_bounds_a_claim_it_can_actually_bound():
    assert fence._keep_alive_seconds('120s') == 120
    assert fence._keep_alive_seconds('5m') == 300
    assert fence._keep_alive_seconds('1h') == 3600
    assert fence._keep_alive_seconds(90) == 90
    # No claim is written for these: an immediate unload, an unbounded
    # residency, or a form this fence will not guess at.
    assert fence._keep_alive_seconds(0) is None
    assert fence._keep_alive_seconds('-1') is None
    assert fence._keep_alive_seconds('forever') is None
    assert fence._keep_alive_seconds(None) is None
    assert fence._keep_alive_seconds(True) is None


def test_expires_at_parses_ollamas_nanosecond_timestamps():
    from datetime import datetime, timezone
    parsed = fence._parse_expires_at('2026-08-02T10:00:00.123456789Z')
    assert parsed == datetime(2026, 8, 2, 10, 0, 0, 123456, tzinfo=timezone.utc).timestamp()
    assert fence._parse_expires_at('2026-08-02T10:00:00+02:00') is not None
    assert fence._parse_expires_at('not a date') is None
    assert fence._parse_expires_at(None) is None


def test_restarted_lds_readopts_the_model_it_loaded_before_the_restart(fence_data_dir):
    endpoint = 'http://127.0.0.1:11434'
    with patch.object(fence.requests, 'get', return_value=_ps()):
        assert fence.mark_before_generate(endpoint, 'lds-model', keep_alive='120s') == 'local'

    fence.reset_for_tests()  # the restart: process memory is gone, the claim is not

    with patch.object(fence.requests, 'get',
                      return_value=_ps_with_expiry('lds-model', _iso(90))), \
            patch.object(fence.requests, 'post') as post:
        assert fence.mark_before_generate(endpoint, 'lds-model', keep_alive='120s') == 'local'
    post.assert_not_called()


def test_a_model_loaded_by_another_app_after_the_restart_is_never_readopted(fence_data_dir):
    """Same name, later residency: the claim must lose."""
    endpoint = 'http://127.0.0.1:11434'
    with patch.object(fence.requests, 'get', return_value=_ps()):
        assert fence.mark_before_generate(endpoint, 'lds-model', keep_alive='120s') == 'local'
    fence.reset_for_tests()

    # Another tool loaded the same model with its own, longer keep-alive: the
    # residency now ends well past anything LDS asked for.
    with patch.object(fence.requests, 'get',
                      return_value=_ps_with_expiry('lds-model', _iso(3000))), \
            patch.object(fence.requests, 'post') as post:
        assert fence.mark_before_generate(endpoint, 'lds-model', keep_alive='120s') == 'blocked'
    post.assert_not_called()


def test_a_claim_that_has_run_out_stops_speaking_for_the_runner(fence_data_dir):
    import json
    import time
    endpoint = 'http://127.0.0.1:11434'
    with patch.object(fence.requests, 'get', return_value=_ps()):
        assert fence.mark_before_generate(endpoint, 'lds-model', keep_alive='120s') == 'local'
    fence.reset_for_tests()

    path = fence._claims_path()
    data = json.loads(path.read_text(encoding='utf-8'))
    data['claims'][endpoint]['lds-model']['deadline'] = time.time() - 600
    path.write_text(json.dumps(data), encoding='utf-8')

    with patch.object(fence.requests, 'get',
                      return_value=_ps_with_expiry('lds-model', _iso(60))):
        assert fence.mark_before_generate(endpoint, 'lds-model', keep_alive='120s') == 'blocked'


def test_no_claim_is_written_for_a_call_that_unloads_straight_away(fence_data_dir):
    endpoint = 'http://127.0.0.1:11434'
    with patch.object(fence.requests, 'get', return_value=_ps()):
        assert fence.mark_before_generate(endpoint, 'lds-model', keep_alive=0) == 'local'
    assert fence._read_claims() == {}


def test_an_unwritable_data_dir_degrades_to_the_old_in_process_behaviour(monkeypatch):
    endpoint = 'http://127.0.0.1:11434'
    monkeypatch.setattr(fence, '_claims_path', lambda: None)
    with patch.object(fence.requests, 'get', return_value=_ps()):
        assert fence.mark_before_generate(endpoint, 'lds-model', keep_alive='120s') == 'local'
    assert fence._read_claims() == {}


# --- A daemon that is not running is not an empty daemon ----------------------
# Reported by socrasteeze (GitHub #20): the probe mapped a REFUSED connection to
# 'empty', so with Ollama stopped the admission path took the claim branch and
# wrote a keep-warm lease for a model that was never loaded. Its only caller
# never returns the lease, so the phantom claim sat in the claim file for its
# whole keep-alive — long enough for the user's OWN later `ollama run` of the
# same model to be adopted as LDS's and unloaded from under them.

def _refused():
    """A connection-refused error shaped like the one requests actually raises."""
    import errno as _errno
    return fence.requests.exceptions.ConnectionError(
        ConnectionRefusedError(_errno.ECONNREFUSED, 'connection refused'))


def test_a_stopped_ollama_is_told_apart_from_an_empty_one():
    endpoint = 'http://127.0.0.1:11434'
    with patch.object(fence.requests, 'get', side_effect=_refused()):
        assert fence._probe(endpoint)[0] == 'down'
    with patch.object(fence.requests, 'get', return_value=_ps()):
        assert fence._probe(endpoint)[0] == 'empty'
    # Anything else the network can do stays 'unknown' - it proves nothing.
    with patch.object(fence.requests, 'get',
                      side_effect=fence.requests.exceptions.ReadTimeout('slow')):
        assert fence._probe(endpoint)[0] == 'unknown'


def test_no_keep_warm_lease_is_written_for_a_model_that_never_loaded(fence_data_dir):
    endpoint = 'http://127.0.0.1:11434'
    with patch.object(fence.requests, 'get', side_effect=_refused()), \
            patch.object(fence.requests, 'post') as post:
        # The call is still admitted: it will fail on its own connection error,
        # or start the daemon and be admitted for real on the retry.
        assert fence.mark_before_generate(endpoint, 'lds-model', keep_alive='120s') == 'local'
    post.assert_not_called()
    assert fence._read_claims() == {}
    with fence._lock:
        assert fence._owned_models == {}


def test_a_model_the_user_loads_after_a_refused_probe_is_never_adopted(fence_data_dir):
    """The teeth of the bug: LDS must not inherit a residency it never created."""
    endpoint = 'http://127.0.0.1:11434'
    with patch.object(fence.requests, 'get', side_effect=_refused()):
        assert fence.mark_before_generate(endpoint, 'lds-model', keep_alive='120s') == 'local'

    # The user starts Ollama and loads that very model themselves.
    with patch.object(fence.requests, 'get',
                      return_value=_ps_with_expiry('lds-model', _iso(90))), \
            patch.object(fence.requests, 'post') as post:
        assert fence.mark_before_generate(endpoint, 'lds-model', keep_alive='120s') == 'blocked'
    post.assert_not_called()


def test_a_stopped_ollama_never_stands_in_comfyuis_way(fence_data_dir):
    """'down' still proves the GPU is free — the release paths must not regress."""
    endpoint = 'http://127.0.0.1:11434'
    with patch.object(fence, '_configured_local_endpoint', return_value=('local', endpoint)), \
            patch.object(fence.requests, 'get', side_effect=_refused()), \
            patch.object(fence.requests, 'post') as post:
        assert fence.ensure_released_for_comfy() is True
        assert fence.release_owned_models(ollama_url=endpoint) is True
        status = fence.fence_status()
        assert fence.unload_foreign_models()['reason'] == 'already-free'
    post.assert_not_called()
    # And the status says the daemon is not there rather than pretending it
    # answered with an empty runner.
    assert status == {'applies': True, 'blocked': False, 'scope': 'local',
                      'reachable': False, 'models': []}


def test_a_stopped_ollama_clears_a_claim_left_by_a_previous_run(fence_data_dir):
    endpoint = 'http://127.0.0.1:11434'
    with patch.object(fence.requests, 'get', return_value=_ps()):
        assert fence.mark_before_generate(endpoint, 'lds-model', keep_alive='120s') == 'local'
    assert fence._read_claims() != {}
    fence.reset_for_tests()
    with patch.object(fence, '_configured_local_endpoint', return_value=('local', endpoint)), \
            patch.object(fence.requests, 'get', side_effect=_refused()):
        assert fence.ensure_released_for_comfy() is True
    assert fence._read_claims() == {}


# --- Status probe + consented unload -----------------------------------------

def test_fence_status_names_what_is_in_the_way_and_clears_when_it_goes(fence_data_dir):
    endpoint = 'http://127.0.0.1:11434'
    with patch.object(fence, '_configured_local_endpoint', return_value=('local', endpoint)):
        with patch.object(fence.requests, 'get', return_value=_ps('other-app-model')):
            state = fence.fence_status()
        assert state == {'applies': True, 'blocked': True, 'scope': 'local',
                         'reachable': True, 'models': ['other-app-model']}
        # Ollama's own idle unload, or the other app closing: the block ends
        # with nobody clicking anything.
        with patch.object(fence.requests, 'get', return_value=_ps()):
            assert fence.fence_status()['blocked'] is False


def test_fence_status_never_reports_blocked_for_an_unreachable_or_remote_daemon():
    endpoint = 'http://127.0.0.1:11434'
    with patch.object(fence, '_configured_local_endpoint', return_value=('local', endpoint)), \
            patch.object(fence.requests, 'get', return_value=_Response({}, status_code=500)):
        state = fence.fence_status()
    assert state['blocked'] is False and state['reachable'] is False

    with patch.object(fence, '_configured_local_endpoint', return_value=('remote', None)), \
            patch.object(fence.requests, 'get') as get:
        assert fence.fence_status() == {'applies': False, 'blocked': False,
                                        'scope': 'remote', 'models': []}
    get.assert_not_called()


def test_fence_status_does_not_call_its_own_residency_foreign_after_a_restart(fence_data_dir):
    endpoint = 'http://127.0.0.1:11434'
    with patch.object(fence.requests, 'get', return_value=_ps()):
        assert fence.mark_before_generate(endpoint, 'lds-model', keep_alive='120s') == 'local'
    fence.reset_for_tests()
    with patch.object(fence, '_configured_local_endpoint', return_value=('local', endpoint)), \
            patch.object(fence.requests, 'get',
                         return_value=_ps_with_expiry('lds-model', _iso(90))):
        assert fence.fence_status()['blocked'] is False


def test_consented_unload_releases_the_external_model_and_proves_the_runner_empty(fence_data_dir):
    endpoint = 'http://127.0.0.1:11434'
    with patch.object(fence, '_configured_local_endpoint', return_value=('local', endpoint)), \
            patch.object(fence.requests, 'get',
                         side_effect=[_ps('other-app-model'), _ps('other-app-model'), _ps()]), \
            patch.object(fence.requests, 'post', return_value=_Response({})) as post:
        assert fence.mark_before_generate(endpoint, 'lds-model') == 'blocked'
        result = fence.unload_foreign_models()
    assert result == {'ok': True, 'reason': 'unloaded',
                      'unloaded': ['other-app-model'], 'still_loaded': []}
    post.assert_called_once_with(f'{endpoint}/api/generate',
                                 json={'model': 'other-app-model', 'keep_alive': 0},
                                 timeout=(10, 30), allow_redirects=False)
    # And the foreign marker is gone, so the next request is admitted.
    with patch.object(fence.requests, 'get', return_value=_ps()):
        assert fence.mark_before_generate(endpoint, 'lds-model') == 'local'


def test_consented_unload_reports_a_runner_that_did_not_actually_empty(fence_data_dir):
    endpoint = 'http://127.0.0.1:11434'
    with patch.object(fence, '_configured_local_endpoint', return_value=('local', endpoint)), \
            patch.object(fence.requests, 'get',
                         side_effect=[_ps('busy-model'), _ps('busy-model')]), \
            patch.object(fence.requests, 'post', return_value=_Response({})):
        result = fence.unload_foreign_models()
    assert result['ok'] is False and result['reason'] == 'still-loaded'
    assert result['still_loaded'] == ['busy-model']


def test_nothing_short_of_the_consent_route_ever_unloads_a_foreign_model(fence_data_dir):
    """The whole point of the fence: no automatic path evicts someone else's model."""
    endpoint = 'http://127.0.0.1:11434'
    with patch.object(fence, '_configured_local_endpoint', return_value=('local', endpoint)), \
            patch.object(fence.requests, 'get', return_value=_ps('other-app-model')), \
            patch.object(fence.requests, 'post') as post:
        assert fence.mark_before_generate(endpoint, 'lds-model') == 'blocked'
        assert fence.fence_status()['blocked'] is True
        assert fence.ensure_released_for_comfy() is False
        assert fence.release_owned_models(ollama_url=endpoint) is False
    post.assert_not_called()


# --- a fence refusal is PUBLISHED (named hold), cleared, and rate-limited ---


def _kcpp_ps(name='kcpp-model'):
    """KoboldCPP's /api/ps: always one resident model, family self-identified,
    expires_at hardcoded decades away — the exact shape that froze a user's
    queue with no explanation anywhere in the app."""
    return _Response({'models': [
        {'name': name, 'model': f'{name}:latest',
         'expires_at': '2055-06-04T19:06:25.5433636+08:00',
         'details': {'family': 'koboldcpp'}},
    ]})


def test_a_refused_handoff_publishes_why_and_success_clears_it():
    endpoint = 'http://127.0.0.1:5001'
    with patch.object(fence, '_configured_local_endpoint', return_value=('local', endpoint)):
        with patch.object(fence.requests, 'get', return_value=_kcpp_ps()):
            assert fence.ensure_released_for_comfy() is False
        blk = fence.last_block()
        assert blk is not None
        assert blk['reason'] == 'foreign'
        assert blk['models'] == ['kcpp-model']
        assert 'koboldcpp' in blk['families']
        # The runner empties (kcpp closed) -> the next hand-off passes and clears.
        with patch.object(fence.requests, 'get', return_value=_ps()):
            assert fence.ensure_released_for_comfy() is True
        assert fence.last_block() is None


def test_a_stale_refusal_stops_explaining_the_queue():
    """A held queue re-refuses every worker cycle, so a LIVE block is always
    fresh; one from a queue that has since emptied must go silent on its own."""
    endpoint = 'http://127.0.0.1:5001'
    with patch.object(fence, '_configured_local_endpoint', return_value=('local', endpoint)), \
            patch.object(fence.requests, 'get', return_value=_kcpp_ps()):
        assert fence.ensure_released_for_comfy() is False
    assert fence.last_block() is not None
    assert fence.last_block(max_age_s=-1) is None


def test_an_unollama_endpoint_is_a_named_refusal_not_a_silent_one():
    """A daemon that answers /api/ps with something else entirely (404, alien
    JSON) used to refuse with NO log and NO published reason — the completely
    invisible frozen queue."""
    endpoint = 'http://127.0.0.1:5001'
    with patch.object(fence, '_configured_local_endpoint', return_value=('local', endpoint)), \
            patch.object(fence.requests, 'get',
                         return_value=_Response({'nope': 1}, status_code=404)):
        assert fence.ensure_released_for_comfy() is False
    blk = fence.last_block()
    assert blk is not None and blk['reason'] == 'unreachable'


def test_repeated_refusals_warn_once_per_interval(caplog):
    """The worker retries a held queue every second; the per-attempt warning was
    sixty identical lines a minute. One a minute carries the same information."""
    endpoint = 'http://127.0.0.1:5001'
    with patch.object(fence, '_configured_local_endpoint', return_value=('local', endpoint)), \
            patch.object(fence.requests, 'get', return_value=_kcpp_ps()):
        with caplog.at_level('WARNING', logger='app.services.ollama_gpu_fence'):
            for _ in range(5):
                assert fence.ensure_released_for_comfy() is False
    warnings = [r for r in caplog.records if 'stays blocked' in r.getMessage()]
    assert len(warnings) == 1
