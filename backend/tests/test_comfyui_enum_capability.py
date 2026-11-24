"""A graph that pins a widget value the user's ComfyUI doesn't have.

Reported by IndependentProcess0 (Reddit): variations died with a raw ComfyUI
console error — `Value not in list: scheduler: 'beta57' not in [...]` — on an
install where plain Klein generation works fine. The shipped 'improve skin.json'
and 'klein_inpaint.json' pin `"scheduler": "beta57"`, and nothing in the app ever
checked whether the target ComfyUI accepts it.

`beta57` is NOT a newer core scheduler: core ComfyUI has never shipped it. The
RES4LYF node pack appends it to comfy.samplers.SCHEDULER_HANDLERS at import, so a
CORE KSampler accepts it only on an install that loaded that pack — which is why
it works for whoever authored the graph and for nobody else.

The app detects the gap and stops with an actionable sentence. It does NOT swap in
a near-equivalent scheduler: that would change the render, leaving two populations
of users producing different images from identical settings.

No network, no generation: /object_info is a fixture.
"""
import pytest

# The scheduler list the reporter's ComfyUI actually published (from his paste),
# i.e. stock core ComfyUI. `beta57` is absent.
STOCK_SCHEDULERS = ['simple', 'sgm_uniform', 'karras', 'exponential', 'ddim_uniform',
                    'beta', 'normal', 'linear_quadratic', 'kl_optimal']
STOCK_SAMPLERS = ['euler', 'euler_ancestral', 'heun', 'dpmpp_2m', 'ddim', 'uni_pc', 'lcm']


def _object_info(schedulers=STOCK_SCHEDULERS, samplers=STOCK_SAMPLERS):
    """A minimal /object_info in ComfyUI's real shape: a combo input is declared as
    [[choice, choice, ...], {options}] — the list ComfyUI quotes back in its
    "Value not in list" error."""
    return {
        'KSampler': {'input': {'required': {
            'seed': ['INT', {'default': 0}],
            'steps': ['INT', {'default': 20}],
            'sampler_name': [list(samplers), {}],
            'scheduler': [list(schedulers), {}],
            'model': ['MODEL'],
        }}},
        'UNETLoader': {'input': {'required': {
            'unet_name': [['klein.safetensors'], {}],
            'weight_dtype': [['default', 'fp8_e4m3fn', 'fp8_e5m2'], {}],
        }}},
    }


# The shape of the graph that actually fails: a CORE KSampler (so the existing
# missing-NODE preflight sees nothing wrong) carrying a non-core scheduler value.
def _klein_like_workflow(scheduler='beta57'):
    return {
        '77': {'class_type': 'KSampler',
               'inputs': {'seed': 1, 'steps': 20, 'sampler_name': 'euler',
                          'scheduler': scheduler, 'model': ['114', 0]}},
        '114': {'class_type': 'UNETLoader',
                'inputs': {'unet_name': 'klein.safetensors', 'weight_dtype': 'fp8_e4m3fn'}},
    }


@pytest.fixture
def probe(app, monkeypatch):
    """Serve a canned /object_info and count the requests, so the enum check can be
    proven to ride the EXISTING cache instead of adding a probe per generation."""
    from app.utils import comfyui

    class _Resp:
        def __init__(self, payload):
            self._p = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._p

    state = {'payload': _object_info(), 'calls': 0, 'boom': False}

    def fake_get(url, timeout=None):
        state['calls'] += 1
        if state['boom']:
            raise OSError('ComfyUI is not running')
        return _Resp(state['payload'])

    from app.services import klein_edit_helper as keh

    def _reset():
        comfyui.clear_model_caches()
        # Success-only TTLs on the shipped-workflow verdicts: module-global, so a
        # clean verdict cached by one test would silently satisfy the next one.
        keh._enums_ok_until = 0.0
        keh._nodes_ok_until = 0.0

    with app.app_context():
        _reset()
        monkeypatch.setattr(comfyui.requests, 'get', fake_get)
        yield state
        _reset()


# --- detection ------------------------------------------------------------

def test_stock_comfyui_is_told_beta57_is_missing_and_where_it_comes_from(app, probe):
    """RED before the fix: nothing looked at the value, the graph went out as-is and
    ComfyUI answered a 400 the user could only decode by reading ComfyUI's own logs."""
    from app.utils import comfyui
    with app.app_context():
        found = comfyui.unsupported_enum_values(_klein_like_workflow())

    assert [(i['input'], i['value'], i['node_id']) for i in found] == [
        ('scheduler', 'beta57', '77')]
    assert found[0]['class_type'] == 'KSampler'

    message = comfyui.format_unsupported_enums_message(found)
    # Names the value...
    assert 'beta57' in message and 'scheduler' in message
    # ...says where it actually comes from (updating ComfyUI would NOT provide it)...
    assert 'RES4LYF' in message
    assert 'https://github.com/ClownsharkBatwing/RES4LYF' in message
    # ...and states the action.
    assert 'restart ComfyUI' in message
    # Paste-safe: no machine path, no drive letter.
    assert '\\Users\\' not in message and 'C:' not in message


def test_the_job_is_refused_before_queuing_with_that_message(app, probe, monkeypatch):
    """The generation must stop at the app, deterministically, instead of becoming a
    raw ComfyUI 400 — and never reach the network."""
    from app.utils import comfyui
    posted = []
    monkeypatch.setattr(comfyui.requests, 'post',
                        lambda *a, **k: posted.append(a) or pytest.fail('POSTed anyway'))
    monkeypatch.setattr(comfyui, '_ensure_comfyui_before_generation', lambda: (True, 'up'))

    with app.app_context():
        result, error = comfyui.queue_prompt_to_comfyui(_klein_like_workflow(), 'client')

    assert result is None
    assert posted == []
    # Same deterministic tag as a ComfyUI validation 400: the queue fails the job
    # now instead of requalifying it as a transient outage worth retrying.
    assert error.startswith('WORKFLOW_INVALIDE')
    assert 'beta57' in error and 'RES4LYF' in error


def test_a_value_we_cannot_attribute_still_gets_an_honest_message(app, probe):
    """An unknown-origin value must not be dressed up with an invented source or an
    invented version number — it says what's missing and the generic fix."""
    from app.utils import comfyui
    wf = _klein_like_workflow(scheduler='some_future_scheduler')
    with app.app_context():
        message = comfyui.format_unsupported_enums_message(
            comfyui.unsupported_enum_values(wf))
    assert 'some_future_scheduler' in message
    assert 'node pack' in message          # generic "update ComfyUI and its packs"
    assert 'RES4LYF' not in message


# --- no regression for installs that DO have the value --------------------

def test_install_that_has_beta57_still_sends_beta57(app, probe, monkeypatch):
    """The whole point of refusing to substitute: whoever's ComfyUI offers beta57
    keeps getting beta57, byte-identical. No detection, no rewrite, no warning."""
    from app.utils import comfyui
    probe['payload'] = _object_info(schedulers=STOCK_SCHEDULERS + ['beta57'])
    sent = {}

    class _Ok:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {'prompt_id': 'p1'}

    def fake_post(url, json=None, headers=None, timeout=None, **kwargs):
        assert kwargs.get('allow_redirects') is False
        sent.update(json or {})
        return _Ok()

    monkeypatch.setattr(comfyui.requests, 'post', fake_post)
    monkeypatch.setattr(comfyui, '_ensure_comfyui_before_generation', lambda: (True, 'up'))

    wf = _klein_like_workflow()
    with app.app_context():
        assert comfyui.unsupported_enum_values(wf) == []
        result, error = comfyui.queue_prompt_to_comfyui(wf, 'client')

    assert error is None and result == {'prompt_id': 'p1'}
    assert sent['prompt']['77']['inputs']['scheduler'] == 'beta57'   # untouched


def test_file_valued_combos_are_never_flagged(app, probe):
    """A model that isn't on disk is NOT a capability gap: it has its own named
    errors and its own fix, and no ComfyUI update produces it. Flagging it here
    would drown the real signal in "your ComfyUI can't do X"."""
    from app.utils import comfyui
    wf = _klein_like_workflow(scheduler='simple')
    wf['114']['inputs']['unet_name'] = 'a_model_the_user_has_not_downloaded.safetensors'
    with app.app_context():
        assert comfyui.unsupported_enum_values(wf) == []


def test_missing_node_class_is_left_to_the_node_preflight(app, probe):
    """A class_type absent from /object_info means "install the pack", which the
    node preflight already says with the right words. Saying it twice, in two
    vocabularies, is noise."""
    from app.utils import comfyui
    wf = {'9': {'class_type': 'SomeCustomNode', 'inputs': {'scheduler': 'beta57'}}}
    with app.app_context():
        assert comfyui.unsupported_enum_values(wf) == []


# --- unreachable probe: don't block, don't invent -------------------------

def test_unreachable_object_info_never_blocks_a_working_install(app, probe, monkeypatch):
    """"Design for EVERY install" cuts both ways: a ComfyUI that answers nothing
    must not be judged. We can't verify, so we don't claim, and the graph goes out
    exactly as before — the pre-existing ComfyUI 400 path stays the backstop."""
    from app.utils import comfyui
    probe['boom'] = True
    sent = {}

    class _Ok:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {'prompt_id': 'p2'}

    def fake_post(url, json=None, headers=None, timeout=None, **kwargs):
        assert kwargs.get('allow_redirects') is False
        sent.update(json or {})
        return _Ok()

    monkeypatch.setattr(comfyui.requests, 'post', fake_post)
    monkeypatch.setattr(comfyui, '_ensure_comfyui_before_generation', lambda: (True, 'up'))

    wf = _klein_like_workflow()
    with app.app_context():
        assert comfyui.fetch_object_info_enums() is None    # None, not {} — unknown
        assert comfyui.unsupported_enum_values(wf) == []
        result, error = comfyui.queue_prompt_to_comfyui(wf, 'client')

    assert error is None and result == {'prompt_id': 'p2'}
    assert sent['prompt']['77']['inputs']['scheduler'] == 'beta57'


# --- cost: one cached probe, not one per generation -----------------------

def test_enum_view_rides_the_existing_object_info_cache(app, probe):
    """/object_info is the heaviest probe in the app (8.8 MB / ~4.8 s measured on a
    node-rich install). The enum check must add ZERO requests: same fetch, same TTL
    cache as the class view, and the refresh-models path drops both."""
    from app.utils import comfyui
    with app.app_context():
        assert comfyui.fetch_object_info_classes() == {'KSampler', 'UNETLoader'}
        assert 'beta57' not in comfyui.fetch_object_info_enums()['KSampler']['scheduler']
        comfyui.unsupported_enum_values(_klein_like_workflow())
        assert probe['calls'] == 1                 # one payload served three readers

        comfyui.clear_model_caches()
        assert comfyui.fetch_object_info_enums() is not None
        assert probe['calls'] == 2                 # dropped -> refetched on demand


def test_only_capability_inputs_are_distilled(app, probe):
    """Keeping the file-name combos out of the distilled map is what makes caching
    it affordable — those arrays are the bulk of the payload."""
    from app.utils import comfyui
    with app.app_context():
        enums = comfyui.fetch_object_info_enums()
    assert set(enums['KSampler']) == {'sampler_name', 'scheduler'}
    assert set(enums['UNETLoader']) == {'weight_dtype'}      # unet_name dropped


# --- the same check, one screen earlier -----------------------------------

def test_setup_preflight_reports_the_gap_before_any_generation(app, probe):
    """Since nothing is substituted, the user MUST learn this before launching.

    The workflow is INJECTED here rather than read from disk: the shipped file is
    now clean, and `test_workflow_portability.py` is what keeps it that way. This
    test owns the other half — that when a graph does carry an unsupported value,
    the Setup screen learns about it instead of the user's first batch."""
    from app.services import klein_edit_helper as keh
    with app.app_context():
        found = keh.klein_unsupported_enums(_klein_like_workflow())
    assert [(i['input'], i['value']) for i in found] == [('scheduler', 'beta57')]
    assert found[0]['pack'] == 'RES4LYF'

    # And the SHIPPED graph, on that same stock ComfyUI, has nothing to report —
    # the fix, seen from the preflight.
    with app.app_context():
        assert keh.klein_unsupported_enums() == []

    probe['boom'] = True
    from app.utils import comfyui
    with app.app_context():
        comfyui.clear_model_caches()
        assert keh.klein_unsupported_enums(_klein_like_workflow()) == []  # fail-open


def test_a_clean_verdict_does_not_re_download_object_info_every_probe(app, probe):
    """The capabilities probe calls this every 30 s and /object_info is ~8.8 MB. Its
    own cache is only 60 s, so without a success TTL here the app would re-pull the
    whole payload every minute, forever, on a machine that is perfectly fine.

    Only a CLEAN verdict is cached: a real gap must clear the moment the user
    installs the pack and restarts ComfyUI, not 5 minutes later."""
    from app.services import klein_edit_helper as keh
    from app.utils import comfyui
    with app.app_context():
        assert keh.klein_unsupported_enums() == []
        before = probe['calls']
        for _ in range(5):
            assert keh.klein_unsupported_enums() == []
        assert probe['calls'] == before          # served by the success TTL

        # A gap, by contrast, is re-probed every time.
        keh._enums_ok_until = 0.0
        comfyui.clear_model_caches()
        assert keh.klein_unsupported_enums(_klein_like_workflow())
        n = probe['calls']
        comfyui.clear_model_caches()
        assert keh.klein_unsupported_enums(_klein_like_workflow())
        assert probe['calls'] > n


# --- Krea 2 Edit: the same net, one engine over ---------------------------------

def _krea_object_info(clip_types):
    """Stock /object_info plus a CLIPLoader whose `type` list is `clip_types`."""
    info = _object_info()
    info['CLIPLoader'] = {'input': {'required': {
        'clip_name': [['qwen.safetensors'], {}],
        'type': [list(clip_types), {}],
        'device': [['default', 'cpu'], {}],
    }}}
    return info


def test_krea_readiness_names_a_comfyui_too_old_for_its_text_encoder(app, probe):
    """The Krea graph loads its Qwen3-VL encoder with CLIPLoader type='krea2'. A
    ComfyUI older than Krea 2 support passes the node check (CLIPLoader is core)
    and then refused the first edit; readiness now says so first."""
    from app.services import krea_edit_helper as krh
    from app.utils import comfyui
    with app.app_context():
        krh._enums_ok_until = 0.0
        probe['payload'] = _krea_object_info(['stable_diffusion', 'qwen_image', 'flux2'])
        found = krh.krea_unsupported_enums()
        assert [(i['class_type'], i['input'], i['value']) for i in found] == [('CLIPLoader', 'type', 'krea2')]
        assert krh._enums_ok_until == 0.0, 'a gap is never cached'
        probe['payload'] = _krea_object_info(['stable_diffusion', 'qwen_image', 'krea2'])
        comfyui.clear_model_caches()
        assert krh.krea_unsupported_enums() == []


def test_krea_enum_probe_fails_open_and_never_caches_an_unread_answer(app, probe):
    from app.services import krea_edit_helper as krh
    from app.utils import comfyui
    with app.app_context():
        krh._enums_ok_until = 0.0
        comfyui.clear_model_caches()
        probe['boom'] = True
        assert krh.krea_unsupported_enums() == []
        assert krh._enums_ok_until == 0.0


def test_an_unsupported_krea_value_keeps_the_engine_dark(monkeypatch):
    from app import capabilities
    from app.services import krea_edit_helper as krh
    gap = {'node_id': '2', 'class_type': 'CLIPLoader', 'input': 'type', 'value': 'krea2'}
    for name, value in (('krea_missing_assets', []), ('krea_missing_nodes', []),
                        ('krea_node_pack_installed', True), ('krea_invalid_assets', []),
                        ('krea_pin_gaps', []), ('resolve_krea_unet', 'krea/base.safetensors'),
                        ('krea_unsupported_enums', [gap])):
        monkeypatch.setattr(krh, name, lambda *a, _v=value, **k: _v)
    *_, ready, unsupported = capabilities._probe_krea({'ok': True})
    assert ready is False and unsupported == [gap]
    monkeypatch.setattr(krh, 'krea_unsupported_enums', lambda: [])
    *_, ready, unsupported = capabilities._probe_krea({'ok': True})
    assert ready is True and unsupported == []
