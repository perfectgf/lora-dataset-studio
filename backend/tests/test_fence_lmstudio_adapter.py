"""The GPU fence, once it has two providers to guard.

The fence is fail-CLOSED: when it cannot prove the card is free, it refuses to
hand it to ComfyUI. Adding a provider is therefore not a matter of teaching it a
second URL — it is a matter of not losing that property along the way. An earlier
draft of this work proposed the adapter return a plain list of resident names,
which cannot distinguish "nothing is loaded" from "I could not tell", and a server
answering only the OpenAI-compatible surface reports no residency at all. Under
that design an LM Studio holding 3 GB of VRAM would have read as a free GPU.

So the tests that matter here are the two that would pass under a broken design
only by accident: `unknown` never collapsing into `empty`, and an endpoint keeping
the wire format it was admitted with after the global provider setting changes.
"""
import pytest

from app import config
from app.services import ollama_gpu_fence as fence
from app.services import vision_lmstudio

pytestmark = pytest.mark.ollama_fence      # these drive the real _probe on purpose

LMS_URL = 'http://127.0.0.1:1299'
OLL_URL = 'http://127.0.0.1:11499'


@pytest.fixture(autouse=True)
def _clean_fence():
    fence._owned_models.clear()
    fence._endpoint_driver.clear()
    yield
    fence._owned_models.clear()
    fence._endpoint_driver.clear()


def _as_lmstudio(app):
    config.save_config({'local_llm': {'provider': 'lmstudio'},
                        'lmstudio': {'url': LMS_URL}})


def test_the_fence_watches_the_active_providers_endpoint(app):
    """Reading ollama.url under an LM Studio install would leave the fence
    guarding a daemon nobody uses while ignoring the one holding the card."""
    with app.app_context():
        _as_lmstudio(app)
        scope, endpoint = fence._configured_local_endpoint()
    assert (scope, endpoint) == ('local', LMS_URL)


def test_the_default_install_still_watches_ollama(app):
    with app.app_context():
        config.save_config({'local_llm': {'provider': 'ollama'},
                            'ollama': {'url': OLL_URL}})
        scope, endpoint = fence._configured_local_endpoint()
    assert (scope, endpoint) == ('local', OLL_URL)


def test_an_endpoint_keeps_the_wire_format_it_was_admitted_with(app):
    """`_owned_models` is keyed by ENDPOINT and the release path walks every key,
    so after a provider switch it holds both. Choosing the driver from the global
    setting would send one of them an unload in the other's format — a request
    that answers 200 and frees nothing, which the fence would read as success."""
    with app.app_context():
        _as_lmstudio(app)
        fence._remember_driver(LMS_URL, 'lmstudio')
        fence._remember_driver(OLL_URL, 'ollama')
        # the user switches back to Ollama mid-session
        config.save_config({'local_llm': {'provider': 'ollama'}})
        assert fence._driver_for(LMS_URL) == 'lmstudio'
        assert fence._driver_for(OLL_URL) == 'ollama'


def test_an_unrecorded_endpoint_defaults_to_ollama(app):
    """Every endpoint this map held before there was a second provider was one."""
    with app.app_context():
        config.save_config({'local_llm': {'provider': 'ollama'}})
        assert fence._driver_for('http://127.0.0.1:9999') == 'ollama'


def test_a_loaded_lmstudio_model_is_seen_as_resident(app, monkeypatch):
    monkeypatch.setattr(vision_lmstudio, 'probe_resident',
                        lambda ep: ('models', ['qwen/qwen3-vl-4b'], {}))
    with app.app_context():
        _as_lmstudio(app)
        fence._remember_driver(LMS_URL, 'lmstudio')
        state, names, expiry = fence._probe(LMS_URL)
    assert state == 'models'
    assert names == {'qwen/qwen3-vl-4b'}
    # LM Studio has no per-model TTL, so there is no expiry to report. The claim
    # logic already tolerates that: a missing value leaves a claim to be judged
    # on its own freshness rather than trusted blindly.
    assert expiry == {}


def test_a_server_that_cannot_report_residency_keeps_the_fence_shut(app, monkeypatch):
    """THE test. A 200 from a server that reports no residency (the OpenAI
    surface, a proxy, another product on the port) must stay `unknown`. Flattened
    to `empty` it reads as a free GPU, and ComfyUI is handed a card LM Studio is
    still holding — the exact inversion this module exists to prevent."""
    monkeypatch.setattr(vision_lmstudio, 'probe_resident',
                        lambda ep: ('unknown', [], {'reason': 'no residency reported'}))
    with app.app_context():
        _as_lmstudio(app)
        fence._remember_driver(LMS_URL, 'lmstudio')
        state, names, _ = fence._probe(LMS_URL)
    assert state == 'unknown', 'a fail-closed guard must not read "cannot tell" as "free"'
    assert names == set()
    # And the admission path refuses on it, rather than loading a second model
    # onto a card whose occupancy it could not establish.
    with app.app_context():
        assert fence.mark_before_generate(LMS_URL, 'qwen/qwen3-vl-4b',
                                          provider='lmstudio') == 'blocked'


def test_nothing_listening_is_down_not_unknown(app, monkeypatch):
    """`down` and `unknown` both mean "no proof of residency", but only `down`
    means the GPU is free — the release paths rely on telling them apart."""
    monkeypatch.setattr(vision_lmstudio, 'probe_resident',
                        lambda ep: ('down', [], {'reason': 'no answer'}))
    with app.app_context():
        _as_lmstudio(app)
        fence._remember_driver(LMS_URL, 'lmstudio')
        assert fence._probe(LMS_URL)[0] == 'down'


def test_releasing_an_lmstudio_endpoint_speaks_lm_studios_api(app, monkeypatch):
    """Not Ollama's `/api/generate {keep_alive: 0}` — which LM Studio does not
    implement, and which would fail quietly while the model stayed resident."""
    called = {}

    def _release(endpoint, model):
        called['args'] = (endpoint, model)
        return True

    monkeypatch.setattr(vision_lmstudio, 'release', _release)
    monkeypatch.setattr(fence.requests, 'post',
                        lambda *a, **kw: pytest.fail('the Ollama unload path was used'))
    with app.app_context():
        _as_lmstudio(app)
        fence._remember_driver(LMS_URL, 'lmstudio')
        assert fence._post_unload(LMS_URL, 'qwen/qwen3-vl-4b') is True
    assert called['args'] == (LMS_URL, 'qwen/qwen3-vl-4b')


def test_a_preloaded_lmstudio_model_is_adopted_not_treated_as_a_stranger(app, monkeypatch):
    """Found by running it, not by reading it: the first real Enhance against a live
    LM Studio came back 409.

    LM Studio inverts what a resident model MEANS. Ollama loads on demand, so a
    model LDS did not load is evidence of another app competing for the card. LM
    Studio has no JIT loading by default — the user loads the model in its app, on
    purpose, so that something can use it, and that something is this. Reading it
    the Ollama way refused EVERY call and made the provider unusable.
    """
    monkeypatch.setattr(vision_lmstudio, 'probe_resident',
                        lambda ep: ('models', ['qwen/qwen3-vl-4b'], {}))
    with app.app_context():
        _as_lmstudio(app)
        scope = fence.mark_before_generate(LMS_URL, 'qwen/qwen3-vl-4b', provider='lmstudio')
    assert scope == 'local'
    # BORROWED, not owned. The two rights are not the same: LDS may use this model
    # because the user loaded it on purpose, and may never unload it for the same
    # reason. Writing it into _owned_models would hand LDS the right to free
    # somebody else's VRAM — "over-adopting is the dangerous direction".
    assert 'qwen/qwen3-vl-4b' in fence._borrowed_models[LMS_URL]
    assert 'qwen/qwen3-vl-4b' not in fence._owned_models.get(LMS_URL, set())


def test_an_empty_lmstudio_is_never_claimed_as_ldss_own(app, monkeypatch):
    """Reported from a real install: "why do I have to keep loading a model?"

    LM Studio reachable with NOTHING loaded is state `empty`. The Ollama reading of
    that is "the card is free, and the model I am about to name will be MINE",
    because Ollama loads on demand. LM Studio loads nothing on LDS's behalf, so the
    claim is about a model LDS will never have loaded — and the moment the user
    loads it by hand, the keep-warm lease unloads it as its own. They load it again,
    and the loop repeats. That is the report.

    The `down` branch already refuses to write a claim for exactly this reason, in
    almost these words. `empty` needs the same refusal, under this provider only.
    """
    monkeypatch.setattr(vision_lmstudio, 'probe_resident', lambda ep: ('empty', [], {}))
    with app.app_context():
        _as_lmstudio(app)
        scope = fence.mark_before_generate(LMS_URL, 'qwen/qwen3-vl-4b',
                                           provider='lmstudio', keep_alive='120s')
    # The call still goes through — it fails on its own with LM Studio's "no models
    # loaded", which names the gesture that fixes it. What must NOT happen is a claim.
    assert scope == 'local'
    assert 'qwen/qwen3-vl-4b' not in fence._owned_models.get(LMS_URL, set()), (
        'LDS claimed a model it cannot have loaded; the keep-warm will unload the '
        "user's own copy of it")
    assert fence._read_claims() == {}, (
        'a persisted claim survives a restart and re-adopts the model as LDS own')


def test_an_empty_ollama_is_still_claimed_because_it_loads_on_demand(app, monkeypatch):
    """The other half, so the fix above is a PROVIDER rule and not a blanket one.

    Ollama loads the model it is asked for. A claim written against an empty
    runner is therefore a claim on LDS's own residency-to-be, and dropping it
    would leak a warm model past every hand-off.
    """
    from app.services import vision_ollama
    monkeypatch.setattr(vision_ollama, 'probe_resident', lambda ep: ('empty', [], {}),
                        raising=False)
    monkeypatch.setattr(fence, '_probe', lambda ep: ('empty', set(), {}))
    with app.app_context():
        config.save_config({'local_llm': {'provider': 'ollama'},
                            'ollama': {'url': OLL_URL}})
        scope = fence.mark_before_generate(OLL_URL, 'qwen3-vl:8b', keep_alive='120s')
    assert scope == 'local'
    assert 'qwen3-vl:8b' in fence._owned_models.get(OLL_URL, set())


def test_a_model_that_is_not_even_loaded_is_still_refused(app, monkeypatch):
    """Borrowing only covers a model that is ALREADY resident. Asking for one that
    is not means LDS would have to load it beside a stranger's — the VRAM fight
    this module exists to prevent."""
    monkeypatch.setattr(vision_lmstudio, 'probe_resident',
                        lambda ep: ('models', ['somebody/else-70b'], {}))
    with app.app_context():
        _as_lmstudio(app)
        scope = fence.mark_before_generate(LMS_URL, 'qwen/qwen3-vl-4b', provider='lmstudio')
    assert scope == 'blocked'


def test_a_co_tenant_model_does_not_block_captioning(app, monkeypatch):
    """LM Studio routinely holds a second model — an embedding one for its own
    document chat, or a chat model beside the VLM. Using a model that is already
    there costs the card nothing, so a co-tenant is not a reason to refuse. Read
    the Ollama way it blocked every Describe and Enhance AND froze the ComfyUI
    queue, telling the user to unload a model they legitimately need."""
    monkeypatch.setattr(vision_lmstudio, 'probe_resident',
                        lambda ep: ('models', ['qwen/qwen3-vl-4b', 'nomic-embed'], {}))
    with app.app_context():
        _as_lmstudio(app)
        assert fence.mark_before_generate(
            LMS_URL, 'qwen/qwen3-vl-4b', provider='lmstudio') == 'local'
    # ...and the endpoint is NOT poisoned for the release path by that admission.
    assert LMS_URL not in fence._foreign_local_endpoints


def test_a_borrowed_model_is_refused_to_comfyui_rather_than_unloaded(app, monkeypatch):
    """The release half stays strict, and stays HONEST: LDS does not free VRAM it
    did not allocate. The queue dock already carries the two consent-gated doors
    for this refusal (unload it for me / share the GPU anyway)."""
    monkeypatch.setattr(vision_lmstudio, 'probe_resident',
                        lambda ep: ('models', ['qwen/qwen3-vl-4b'], {}))
    monkeypatch.setattr(vision_lmstudio, 'release',
                        lambda *a, **kw: pytest.fail('a borrowed model was unloaded'))
    with app.app_context():
        _as_lmstudio(app)
        fence.mark_before_generate(LMS_URL, 'qwen/qwen3-vl-4b', provider='lmstudio')
        assert fence._release_endpoint(LMS_URL, set()) is False


def test_the_refusal_names_the_server_the_user_actually_runs(app):
    """An accurate refusal about the wrong product is a support ticket, not a fix."""
    with app.app_context():
        _as_lmstudio(app)
        assert 'LM Studio' in fence.blocked_message()
        config.save_config({'local_llm': {'provider': 'ollama'}})
        assert fence.blocked_message() == fence.FENCE_BLOCKED_MESSAGE
