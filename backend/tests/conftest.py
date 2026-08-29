import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _basetemp_guard


def pytest_configure(config):
    """Refuse a --basetemp another live pytest run already owns.

    pytest rm_rf's the basetemp it is given (see _basetemp_guard), so two runs
    sharing one wipe each other's tmp_path trees mid-flight and fail on things
    that have nothing to do with the code. Several agents running this suite in
    parallel is now the norm, so the collision is refused here — with a message
    naming the other run — instead of surfacing as a phantom failure plus a
    handful of "ERROR at setup" half a suite later."""
    basetemp = getattr(config.option, 'basetemp', None)
    if not basetemp:
        return                        # no --basetemp: pytest numbers per run, safe
    conflict = _basetemp_guard.claim(basetemp)
    if conflict:
        pytest.exit(conflict, returncode=pytest.ExitCode.USAGE_ERROR)
    # Ours is claimed - now collect what dead runs left next to it. A
    # killed pytest never releases: ~45 stale basetemps once held 105 GB
    # and made the training preflight refuse launches like a regression.
    _basetemp_guard.sweep_stale_siblings(basetemp)


def pytest_unconfigure(config):
    basetemp = getattr(config.option, 'basetemp', None)
    if basetemp:
        _basetemp_guard.release(basetemp)

@pytest.fixture(autouse=True)
def _restore_secret_env():
    """set_secrets() writes os.environ directly; snapshot & restore the secret keys.

    Also CLEAR them at setup: config.py runs load_dotenv(ENV_PATH) at import, and at
    collection time (before LDS_ENV is pointed at a tmp file) ENV_PATH is the real
    repo .env — so a developer who saved a real Gemini/OpenAI key via the app would
    leak it into os.environ and make "unconfigured" probes see a key. Starting each
    test with the keys unset makes the suite independent of the local .env; tests
    that need a key set it themselves via monkeypatch.setenv."""
    import os
    from app.config import SECRET_KEYS as keys   # stays in sync as keys are added
    saved = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ.pop(k, None)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

@pytest.fixture(autouse=True)
def _no_live_comfyui_vram_release(monkeypatch):
    """Every GPU-exclusive vision window POSTs /free to ComfyUI — for real.

    `gpu_exclusive_vision_window` calls `free_comfyui_vram()`, which is a live
    `requests.post('<comfyui>/free', {"unload_models": true, "free_memory": true})`
    with a 10 s timeout. Nothing mocks it, so on a developer machine where
    ComfyUI is running the unit suite REALLY unloads that ComfyUI's models —
    measured 2026-07-28: eleven test files reach 127.0.0.1:8188, among them the
    whole bank-vision-concurrency file.

    Two costs, and the second is the one that gets misfiled. It has a side
    effect on a program nobody asked it to touch; and its duration is whatever
    that program happens to be doing — instant when ComfyUI is idle, seconds
    when it is mid-generation, the full 10 s timeout when the port answers but
    the server is wedged. A test asserting "Stop returns in under 5 s" then
    passes or fails on the state of an unrelated process. That is not a flake,
    it is an undeclared dependency.

    Tests that are ABOUT this call (test_vision_features) monkeypatch it
    themselves; a later setattr wins over this one, so they are unaffected."""
    from app.utils.comfyui import ComfyVramFreeVerdict
    monkeypatch.setattr('app.utils.comfyui.free_comfyui_vram',
                        lambda *a, **k: ComfyVramFreeVerdict.FREED)


@pytest.fixture(autouse=True)
def _reset_inmemory_registries():
    """dataset_activity is a process-global in-memory store (a batch dies with the
    process, not the request). With :memory: DBs each test restarts dataset ids at
    1, so a batch a PRIOR test began on 'dataset 1' would look live to the next
    test's fresh 'dataset 1' — enough to make the kind-switch guard 409 spuriously.
    Clear it around every test so in-memory activity never leaks across cases.

    The bank folder-sync cooldowns are process-global for the same reason: bank
    id 1 of a prior test would make the next test's first walk a no-op.

    The vision keep-warm LEASE is the same kind of global, and the nastiest: any
    test that crops/uploads through detect_head_bbox grants a 120 s lease, and
    every later test whose code path is "about to take the GPU" then calls
    revoke() -> a REAL HTTP POST to whatever answers on the Ollama URL. That is
    how test_training_queue_atomic failed once in a full suite and never alone:
    the launch under test paid for a live unload of a machine's actual Ollama,
    which can take tens of seconds. The suite must never depend on a lease left
    behind by an earlier test, nor talk to a live Ollama by accident.

    The 🔤 text-search query cache is a third one, and it bites the same way in
    reverse: it is keyed by PHRASE only (a CLIP vector depends on the checkpoint,
    not on the bank), so a query encoded by one test would be served from memory
    to the next — whose LDS_DATA_DIR is a different empty tmp dir. A test proving
    "the encoder is invoked exactly once" would then see zero calls and fail, and
    one proving "no ML python ⇒ 503" would silently get a cache hit and a 200.
    Both did, before this line existed."""
    from app.services import bank_jobs, bank_undo, clip_text_encoder
    from app.services import dataset_activity
    from app.services import image_bank_service, ollama_gpu_fence, vision_keepalive
    dataset_activity.reset()
    bank_jobs.reset()
    bank_undo.reset()
    image_bank_service.reset_folder_sync()
    image_bank_service.reset_score_memo()
    vision_keepalive.forget_lease()
    ollama_gpu_fence.reset_for_tests()
    clip_text_encoder.forget_memory_cache()
    yield
    dataset_activity.reset()
    bank_jobs.reset()
    bank_undo.reset()
    image_bank_service.reset_folder_sync()
    image_bank_service.reset_score_memo()
    vision_keepalive.forget_lease()
    ollama_gpu_fence.reset_for_tests()
    clip_text_encoder.forget_memory_cache()
    clip_text_encoder.release()

@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv('LDS_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('LDS_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setenv('LDS_ENV', str(tmp_path / '.env'))
    # A developer may have real extensions cloned into backend/extensions/;
    # the suite must never load them. Point the loader at an empty dir.
    monkeypatch.setenv('LDS_EXTENSIONS_DIR', str(tmp_path / 'no-extensions'))
    import app.config as _cfg
    monkeypatch.setattr(_cfg, 'ENV_PATH', tmp_path / '.env')   # never touch the real .env in tests
    # config.py caches load_config() in a module-level global keyed on nothing but
    # "has it been loaded before" -- it isn't tied to LDS_CONFIG. Without resetting it
    # here, a test that calls save_config() with a real comfyui.base_dir leaks that
    # value into every later test's "fresh" app (same process, stale cache), even
    # though each test gets its own tmp_path/env vars. Task 14 (Klein path) hit this:
    # a test asserting "ComfyUI unconfigured -> RuntimeError" silently inherited a
    # previous test's real base_dir and passed for the wrong reason.
    monkeypatch.setattr(_cfg, '_cache', None)
    # capabilities.py caches its WHOLE probe for 30 s in a module global, and that
    # clock does not know tests exist: a file that ran seconds earlier -- with its
    # own tmp config, so legitimately seeing no ai-toolkit -- leaves 'aitoolkit
    # invalid' in the cache, and the next test's `client.get(.../preflight)` gets
    # 409 instead of 200 no matter what config IT wrote. Reproduced 2/2 by running
    # test_masked_dataset_setting.py before test_training_preflight.py.
    #
    # Same class as the two bank tests that failed a release this morning: a test
    # whose answer depends on a shared clock fails INTERMITTENTLY, and intermittent
    # reads as random. Clearing both caches per test makes the suite say what the
    # test set up, and nothing else.
    import app.capabilities as _caps
    monkeypatch.setattr(_caps, '_cache', None)
    monkeypatch.setattr(_caps, '_cache_ts', 0.0)
    _caps._import_cache.clear()
    from app import create_app
    application = create_app({'TESTING': True, 'WTF_CSRF_ENABLED': False,
                              'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:'})
    yield application
    # Hand every pooled sqlite connection back. Without this each test
    # leaked its in-memory engine, and the suite drowned in ~27 900
    # ResourceWarnings (~93 % of all warnings) - unable to signal a NEW
    # warning over the noise.
    from app.extensions import db as _db
    with application.app_context():
        _db.session.remove()
        _db.engine.dispose()

@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture(autouse=True)
def _no_hugging_face_gate_call(request, monkeypatch):
    """Keep the unit suite off the public internet.

    `cloud_training._assert_official_base_reachable` really opens
    https://huggingface.co/api/models/<repo>/tree/main, with an 8 s timeout, and
    **83 tests reached it** — measured, not estimated. Two costs, one of which
    has not been paid yet:

    * duration, and it is the answer to a mystery from 2026-07-28: two agents
      saw the suite take 620 s instead of 505 s and read that as the CAUSE of
      failing tests. It was this, a network dependency nobody declared;
    * a release trap. The call fails OPEN on timeouts and outages, but it RAISES
      on 401/403. The day HF gates a base, changes a quota, or a stale token is
      present, ~83 cloud tests fail at once and it will look exactly like a
      flake storm — the same shape as the ordering bug that failed a release
      that morning.

    A unit test must never depend on someone else's uptime. The tests that exist
    to exercise this gate opt back in with @pytest.mark.hf_gate, and they stub
    urlopen themselves, so the behaviour stays covered — only the traffic goes.
    """
    if request.node.get_closest_marker('hf_gate'):
        return
    from app.services import cloud_training as _ct
    monkeypatch.setattr(_ct, '_assert_official_base_reachable',
                        lambda *a, **k: None, raising=False)


@pytest.fixture(autouse=True)
def _ollama_fence_never_reads_this_machine(request, monkeypatch):
    """Keep the unit suite off THIS machine's Ollama.

    The local-GPU fence probes `/api/ps` on the configured local endpoint before
    every Ollama call, and with no config that endpoint is 127.0.0.1:11434 — the
    developer's real Ollama. So the verdict of ~24 tests across test_captioning
    and test_dataset_service was decided by whatever happened to be resident
    there: empty runner, everything passes; anything loaded outside LDS (the
    image generator, another agent, a model the developer pulled a minute ago)
    and the fence correctly refuses, LocalOllamaFenceError propagates, and they
    all fail at once. Measured 2026-08-02: 0 failures with an empty runner, 24
    with `qwen3-vl:4b-instruct` loaded — same commit, same code, both times.

    That is not a flake, it is an undeclared dependency, and it is the worst
    kind: it makes CI red on a machine state nobody changed on purpose, and the
    failure looks like the fence is broken when the fence is doing its job.

    The claim file is stubbed for the same reason in the other direction: a test
    that admits a model would otherwise leave a claim on disk, and the next test
    to probe the same endpoint could re-adopt from it.

    Tests that are ABOUT the fence opt back in with @pytest.mark.ollama_fence
    and drive `requests` themselves, so the behaviour stays covered — only the
    dependency on someone else's GPU goes.
    """
    from app.services import ollama_gpu_fence as _fence
    if request.node.get_closest_marker('ollama_fence'):
        # Still isolate the process-global registries: ownership must never leak
        # from one test into the next, whoever drives the probe.
        _fence.reset_for_tests()
        yield
        _fence.reset_for_tests()
        return
    monkeypatch.setattr(_fence, '_probe', lambda endpoint: ('empty', set(), {}))
    monkeypatch.setattr(_fence, '_record_claim', lambda *a, **k: None)
    monkeypatch.setattr(_fence, '_read_claims', dict)
    _fence.reset_for_tests()
    yield
    _fence.reset_for_tests()


@pytest.fixture(autouse=True)
def _nothing_else_reads_this_machines_ollama(request, monkeypatch):
    """Keep the unit suite off THIS machine's Ollama — the OTHER two doors.

    The fence guard above closed one door in 2026-08-02 and left the building
    open. Measured 2026-08-29 with a socket tripwire on port 11434, full suite:
    **119 real connections, 59 tests, 24 files**. Two distinct leaks, and they
    do not cost the same thing:

    * `capabilities.probe_ollama` / `probe_ollama_model` — GET /api/tags, 118 of
      them, from every test that builds a capabilities payload (diagnostic,
      cloud routes, krea bases, seedvr2, setup_state, watermark…). A read: it
      lists installed models, loads nothing, reserves no VRAM. Cheap, but it
      still lets a daemon nobody declared decide a verdict.
    * `vision_ollama.describe_image_ollama` — **POST /api/generate**, reached by
      `detect_head_bbox` whenever a test uploads an image (test_dataset_routes,
      test_json_body_strict). That one is not a read: it LOADS the 8 GB vision
      model onto the shared GPU. Through LDS's own GPU fence that can stall a
      running Test Studio queue, and it competes with whatever else holds the
      card. A unit suite must never be able to do that.

    Forcing both to the "no daemon" answer is what CI already sees (no Ollama on
    the runner), so every machine agrees with CI instead of inventing a third
    behaviour. The stubs return exactly what the real failure path returns —
    False / [] / '' — so no caller learns a shape it would not see in the wild.

    `/api/tags` is the discriminator for the probe half on purpose: it is
    Ollama's path and nothing else's — ComfyUI probes `/history` through the
    same `_http_ok` and must keep working.

    Tests that are ABOUT any of this opt back in with @pytest.mark.ollama_http
    and drive `requests` themselves, so the behaviour stays covered.
    """
    if request.node.get_closest_marker('ollama_http'):
        return
    import errno as _errno

    import requests as _requests

    # Cut the SOCKET, not the functions. The first version of this guard
    # stubbed `describe_image_ollama` & co. and turned 30 tests red — because
    # test_captioning (26 of them), test_vision_keepalive and test_studio_service
    # mock `vision_ollama.requests.post` and then exercise the REAL function to
    # assert its parsing, its retries and its lease bookkeeping. Replacing the
    # function stepped over their mock and tested nothing. Refusing the
    # connection underneath leaves every line of app logic running, on the exact
    # path a machine with no daemon takes; a test that installs its own
    # `requests` double still wins, because its setattr comes after this one.
    real_get, real_post = _requests.get, _requests.post

    def _refuse(url):
        # Shaped like the refusal a machine with no daemon actually produces —
        # a ConnectionRefusedError carrying ECONNREFUSED, wrapped by requests.
        # The shape is load-bearing, not decoration: ollama_gpu_fence walks the
        # exception chain (`_connection_refused`) to tell "nothing is listening"
        # from "something went wrong", and a bare ConnectionError would be filed
        # as `unknown` instead of `down`. A guard whose whole argument is "every
        # machine agrees with CI" must not invent a third answer of its own.
        # Same constructor as tests/test_ollama_gpu_fence.py::_refused.
        raise _requests.exceptions.ConnectionError(
            ConnectionRefusedError(
                _errno.ECONNREFUSED,
                f'the unit suite must not reach this machine\'s Ollama ({url}). '
                'A test that means to drive it takes @pytest.mark.ollama_http.'))

    def _get(url, *args, **kwargs):
        if ':11434' in str(url):
            _refuse(url)
        return real_get(url, *args, **kwargs)

    def _post(url, *args, **kwargs):
        if ':11434' in str(url):
            _refuse(url)
        return real_post(url, *args, **kwargs)

    # Port, not path: 11434 IS "the daemon on this machine", while a test that
    # points ollama.url at a fixture server of its own is not talking to it and
    # must keep working. Everything else on `requests` — ComfyUI, Hugging Face,
    # the cloud provider — is delegated untouched.
    _get.lds_ollama_guard = _post.lds_ollama_guard = True
    monkeypatch.setattr(_requests, 'get', _get)
    monkeypatch.setattr(_requests, 'post', _post)
