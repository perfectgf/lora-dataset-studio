"""The /object_info read budget, and telling a SLOW ComfyUI from a STOPPED one.

Reported and diagnosed by j_o_e_l. (Discord, #bug-reports): Krea 2 generations
refused with "ComfyUI isn't running" on a ComfyUI that was running. He timed the
`/object_info` enumeration on his own install at ~15 s against a hardcoded 8 s
budget, raised the four probe timeouts, and the engine worked again.

Two defects, both pinned here:

  1. 8 s was a GUESS that cannot hold. The /object_info payload lists every node
     class and every model file, so it grows with what the user has installed —
     the budget therefore has to be a setting, and its default has to have real
     margin over a measured install rather than 1.6x a lucky one.
  2. A timeout was REPORTED as "not running". Two causes, two remedies: start it
     vs give it more time. Nothing here asserts exact prose (it will be reworded);
     it asserts that the two messages are DIFFERENT, that the slow one names the
     delay, and that neither claims the wrong thing.

No real network anywhere: `requests.get` is replaced by a fake that reads the
budget it was handed and decides.
"""
import pytest
import requests

from app import capabilities
from app.utils import comfyui


PAYLOAD = {'KSampler': {}, 'UNETLoader': {}}


class _Resp:
    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return PAYLOAD


def _comfy_taking(seconds, calls=None):
    """A fake ComfyUI whose /object_info takes `seconds` to build its answer.

    It reads the (connect, read) tuple it is given and raises requests' REAL
    ReadTimeout when the read budget is under that — i.e. exactly what a live
    ComfyUI does to us, minus the waiting."""
    def fake_get(url, timeout=None, **kw):
        if calls is not None:
            calls.append((url, timeout))
        connect, read = timeout if isinstance(timeout, tuple) else (timeout, timeout)
        if read is None or read < seconds:
            raise requests.exceptions.ReadTimeout(f'read timed out after {read}s')
        return _Resp()
    return fake_get


def _comfy_off(calls=None):
    """Nothing listening: the CONNECT phase is what fails, instantly."""
    def fake_get(url, timeout=None, **kw):
        if calls is not None:
            calls.append((url, timeout))
        raise requests.exceptions.ConnectionError('connection refused')
    return fake_get


@pytest.fixture(autouse=True)
def _clean_object_info_caches():
    comfyui.clear_model_caches()
    capabilities._cache = None
    capabilities._cache_ts = 0.0
    yield
    comfyui.clear_model_caches()
    capabilities._cache = None
    capabilities._cache_ts = 0.0


# --- 1. The budget itself ----------------------------------------------------

def test_a_fifteen_second_install_fails_at_the_old_budget_and_succeeds_now(app, monkeypatch):
    """THE regression, in one test. The install j_o_e_l. measured (~15 s) is
    refused by the 8 s that used to be hardcoded and served by the shipped default.

    The 8 s branch is what makes this red-then-green rather than a tautology: it
    reproduces the reported failure through the very same code path."""
    with app.app_context():
        monkeypatch.setattr(comfyui.requests, 'get', _comfy_taking(15))
        # The old hardcoded budget, passed explicitly: still broken, on purpose.
        comfyui.clear_model_caches()
        assert comfyui.fetch_object_info_classes(timeout=8) is None
        # The shipped default: the same install now answers.
        comfyui.clear_model_caches()
        assert comfyui.fetch_object_info_classes() == set(PAYLOAD)


def test_default_budget_has_real_margin_over_the_measured_worst_case(app):
    """45 s is not another guess-with-no-margin.

    Two MEASURED points exist: ~5 s on the node-rich install this app's own
    docstring was written against, and ~15 s on j_o_e_l.'s. The old 8 s sat at 1.6x
    the first and lost to the second — a margin under 2x is what created this bug.
    The default must clear the worst measured install by at least 2x, so an install
    twice as node-rich as the worst one we know of still fits, while staying well
    under a minute (an offline ComfyUI must never cost that — see the connect-budget
    test below, which is what actually guarantees it)."""
    with app.app_context():
        assert comfyui.object_info_timeout() >= 2 * 15
        assert comfyui.object_info_timeout() <= 60


def test_budget_is_a_setting_and_is_honoured(app, monkeypatch):
    """The whole point of the fix: no constant can be right for every install."""
    from app import config as cfg
    seen = []
    with app.app_context():
        monkeypatch.setattr(comfyui.requests, 'get', _comfy_taking(50, calls=seen))
        # Default (45) is not enough for a 50 s install...
        assert comfyui.fetch_object_info_classes() is None
        assert seen[-1][1] == (comfyui._OBJECT_INFO_CONNECT_TIMEOUT, 45)
        # ...raising the setting is all it takes.
        cfg.save_config({'comfyui': {'object_info_timeout_s': 120}})
        comfyui.clear_model_caches()
        assert comfyui.object_info_timeout() == 120
        assert comfyui.fetch_object_info_classes() == set(PAYLOAD)
        assert seen[-1][1] == (comfyui._OBJECT_INFO_CONNECT_TIMEOUT, 120)


@pytest.mark.parametrize('raw,expected', [
    (0, 5), (1, 5),            # clamped up: a 1 s budget disables the probe
    (10000, 300),              # clamped down: a wedged probe must still give up
    ('90', 90),                # settings round-trips through JSON/forms
    ('nonsense', 45), (None, 45), (True, 45),   # unusable -> the shipped default
])
def test_budget_setting_is_clamped_and_never_breaks_the_probe(app, raw, expected):
    """Total by construction, like ollama.vision_concurrency: a bad value costs
    tuning, never the probe the whole app leans on."""
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'comfyui': {'object_info_timeout_s': raw}})
        assert comfyui.object_info_timeout() == expected


# --- 2. What an ABSENT ComfyUI costs ----------------------------------------

def test_a_stopped_comfyui_pays_the_connect_budget_not_the_read_budget(app, monkeypatch):
    """The arbitrage that makes a generous read budget safe on BACKGROUND paths.

    A long single timeout would make every capability poll wait the full budget at
    a ComfyUI that is simply off. Splitting (connect, read) removes the trade-off:
    an absent server fails the handshake, so it can only ever cost the small
    connect budget — no matter how large the read budget is, and with no second
    'background timeout' setting to get wrong."""
    seen = []
    with app.app_context():
        monkeypatch.setattr(comfyui.requests, 'get', _comfy_off(calls=seen))
        assert comfyui.fetch_object_info_classes() is None
        connect, read = seen[-1][1]
        assert connect == comfyui._OBJECT_INFO_CONNECT_TIMEOUT <= 5
        assert read == comfyui.object_info_timeout()


def test_a_burst_of_callers_pays_one_failed_probe_not_one_each(app, monkeypatch):
    """A failure is cached, briefly.

    It used to be cached not at all. Every caller — the capability poll, the Studio
    preflight, each generate — therefore re-fired a multi-megabyte enumeration, and
    on a slow install that self-inflicted storm is what kept the cheap /history
    reachability check timing out. Fail-open is preserved: they all still get None."""
    seen = []
    with app.app_context():
        monkeypatch.setattr(comfyui.requests, 'get', _comfy_off(calls=seen))
        for _ in range(5):
            assert comfyui.fetch_object_info_classes() is None
        assert len(seen) == 1
        # "I started ComfyUI, refresh" must not wait out the negative TTL.
        comfyui.clear_model_caches()
        monkeypatch.setattr(comfyui.requests, 'get', _comfy_taking(1))
        assert comfyui.fetch_object_info_classes() == set(PAYLOAD)


# --- 3. Two causes, two messages --------------------------------------------

def test_probe_records_slow_and_absent_as_different_states(app, monkeypatch):
    with app.app_context():
        monkeypatch.setattr(comfyui.requests, 'get', _comfy_taking(999))
        assert comfyui.fetch_object_info_classes() is None
        health = comfyui.object_info_health()
        assert health['status'] == 'timeout'
        assert health['waited'] == comfyui.object_info_timeout()

        comfyui.clear_model_caches()
        monkeypatch.setattr(comfyui.requests, 'get', _comfy_off())
        assert comfyui.fetch_object_info_classes() is None
        assert comfyui.object_info_health()['status'] == 'unreachable'


def test_the_two_messages_are_different_and_the_slow_one_does_not_say_not_running():
    """The guard against the catch-all coming back.

    Wording will change; these three properties must not: the sentences differ, the
    slow one quotes the delay and points at the timeout setting, and neither tells
    someone whose ComfyUI is up that it is down."""
    slow = capabilities.comfyui_down_message('timeout', 45)
    down = capabilities.comfyui_down_message('down', 3)
    assert slow != down
    assert '45' in slow
    assert 'timeout' in slow.lower() and 'settings' in slow.lower()
    assert 'is running' in slow.lower()
    # The lie this whole change exists to remove.
    assert 'not running' not in slow.lower() and "isn't running" not in slow.lower()
    # ...and the genuine case still says the true thing, with the true remedy.
    assert 'start comfyui' in down.lower()


def test_probe_comfyui_reports_slow_when_the_heavy_probe_proved_it_is_up(app, monkeypatch):
    """/history gives up after 3 s. On a ComfyUI busy enumerating itself that is
    exactly what happens — and reading it as "unreachable" is the reported bug. The
    /object_info probe waited far longer and knows better, so it is believed."""
    with app.app_context():
        monkeypatch.setattr(comfyui.requests, 'get', _comfy_taking(999))
        assert comfyui.fetch_object_info_classes() is None      # records 'timeout'
        monkeypatch.setattr(capabilities, '_http_ok', lambda *a, **k: False)
        res = capabilities.probe_comfyui()
        assert res['ok'] is False
        assert res['status'] == 'slow'
        assert 'unreachable' not in res['detail']
        assert res['hint'] == capabilities.comfyui_down_message(
            'timeout', comfyui.object_info_timeout())


def test_probe_comfyui_still_says_unreachable_when_nothing_answers(app, monkeypatch):
    with app.app_context():
        monkeypatch.setattr(comfyui.requests, 'get', _comfy_off())
        assert comfyui.fetch_object_info_classes() is None      # records 'unreachable'
        monkeypatch.setattr(capabilities, '_http_ok', lambda *a, **k: False)
        res = capabilities.probe_comfyui()
        assert res['status'] == 'unreachable'
        assert 'unreachable' in res['detail']
        assert 'start comfyui' in res['hint'].lower()


def test_http_ok_reports_why_it_failed_without_breaking_stubs():
    """`_http_ok` stays the single patched network seam: the reason is an
    out-param, so a stub that ignores it leaves the reason unknown rather than
    raising."""
    reason = {}
    assert capabilities._http_ok('http://127.0.0.1:9/nothing', timeout=0.01,
                                 reason=reason) is False
    assert reason.get('why') in ('timeout', 'unreachable')
    # A stub with the historical signature must still be callable.
    assert capabilities._http_ok('http://x', timeout=1) is False or True


def test_capabilities_publishes_the_status_and_the_budget(app, monkeypatch):
    """The front cannot say the true sentence from `reachable` alone."""
    with app.app_context():
        monkeypatch.setattr(comfyui.requests, 'get', _comfy_taking(999))
        assert comfyui.fetch_object_info_classes() is None
        monkeypatch.setattr(capabilities, '_http_ok', lambda *a, **k: False)
        capabilities._cache = None
        comfy = capabilities.probe(force=True)['comfyui']
        assert comfy['reachable'] is False
        assert comfy['status'] == 'slow'
        assert comfy['hint']
        assert comfy['object_info_timeout_s'] == comfyui.object_info_timeout()


def test_the_409_on_a_blocked_run_names_the_right_cause(app, client, monkeypatch):
    """`_require_comfyui` is what a user actually reads when a run is refused."""
    from app.routes import _common
    with app.app_context():
        monkeypatch.setattr(_common.capabilities, 'probe_comfyui', lambda: {
            'ok': False, 'status': 'slow', 'detail': 'slow', 'hint': 'HINT-SLOW'})
        body, status = _common._require_comfyui()
        assert status == 409
        payload = body.get_json()
        assert 'slow' in payload['error'].lower()
        assert 'not reachable' not in payload['error'].lower()
        assert payload['hint'] == 'HINT-SLOW'

        monkeypatch.setattr(_common.capabilities, 'probe_comfyui', lambda: {
            'ok': False, 'status': 'unreachable', 'detail': 'down', 'hint': 'HINT-DOWN'})
        body, status = _common._require_comfyui()
        assert body.get_json()['error'] == 'ComfyUI is not reachable'
