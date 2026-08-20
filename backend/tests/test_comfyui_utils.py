"""Tests for the lifted app.utils.comfyui module: the shared trained-LoRA
parser (label + group MUST share one parse — the drift-proof invariant),
config-driven listers (empty/None-safe when ComfyUI isn't configured), and
the LoRA-chain injectors (allowed-whitelist respected)."""
from unittest.mock import MagicMock, patch

from app.utils.comfyui import (
    trained_lora_group, format_trained_lora_label, family_of_lora,
    inject_zimage_loras, _trained_lora_trigger,
)


def _response(payload=None, *, status_code=200):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    return response


def test_cancel_comfyui_prompt_never_interrupts_matching_running_prompt(app):
    from app.utils.comfyui import cancel_comfyui_prompt
    queue = {
        'queue_running': [
            [1, 'other-prompt', {}, {'client_id': 'other-job'}, []],
            [2, 'target-prompt', {}, {'client_id': 'target-job'}, []],
        ],
        'queue_pending': [],
    }
    with app.app_context(), \
         patch('app.utils.comfyui.requests.get', return_value=_response(queue)), \
         patch('app.utils.comfyui.requests.post') as post:
        assert cancel_comfyui_prompt('target-prompt', 'target-job') is False
    post.assert_not_called()


def test_cancel_comfyui_prompt_never_interrupts_unrelated_running_prompt(app):
    from app.utils.comfyui import cancel_comfyui_prompt
    queue = {
        'queue_running': [[1, 'other-prompt', {}, {'client_id': 'other-job'}, []]],
        'queue_pending': [],
    }
    with app.app_context(), \
         patch('app.utils.comfyui.requests.get', return_value=_response(queue)), \
         patch('app.utils.comfyui.requests.post') as post:
        assert cancel_comfyui_prompt('target-prompt', 'target-job') is False
    post.assert_not_called()


def test_cancel_comfyui_prompt_deletes_matching_pending_prompt(app):
    from app.utils.comfyui import cancel_comfyui_prompt
    queue = {
        'queue_running': [],
        'queue_pending': [[1, 'target-prompt', {}, {'client_id': 'target-job'}, []]],
    }
    with app.app_context(), \
         patch('app.utils.comfyui.requests.get', return_value=_response(queue)), \
         patch('app.utils.comfyui.requests.post', return_value=_response({})) as post:
        assert cancel_comfyui_prompt('target-prompt', 'target-job') is True
    assert post.call_args.args[0].endswith('/queue')
    assert post.call_args.kwargs['json'] == {'delete': ['target-prompt']}


def test_history_probe_requires_the_exact_prompt_key(app):
    from app.utils.comfyui import ComfyHistoryHealth, get_comfyui_history_probe

    # A direct-looking entry cannot prove it belongs to this exact prompt. It
    # must pause the scheduler rather than completing unrelated remote GPU work.
    with app.app_context(), patch(
            'app.utils.comfyui.requests.get',
            return_value=_response({'outputs': {}, 'status': {}})):
        probe = get_comfyui_history_probe('expected-prompt')
    assert probe.health is ComfyHistoryHealth.UNHEALTHY
    assert probe.detail == 'history does not contain requested prompt'



def test_comfyui_gpu_decisions_refuse_redirects(app):
    """A proxy redirect is not proof of a prompt, history, queue, or VRAM state."""
    import importlib
    import app.utils.comfyui as comfyui

    # The suite safety fixture replaces /free. Reload just this module so this
    # contract exercises the real request call without touching a live server.
    comfyui = importlib.reload(comfyui)

    with app.app_context(), patch(
            'app.utils.comfyui.requests.post',
            return_value=_response({'prompt_id': 'prompt-1'})) as post:
        queued, error = comfyui.queue_prompt_to_comfyui(
            {'1': {}}, 'client-1', worker_url='http://comfy.local')
    assert error is None and queued['prompt_id'] == 'prompt-1'
    assert post.call_args.kwargs['allow_redirects'] is False

    with app.app_context(), patch(
            'app.utils.comfyui.requests.post',
            return_value=_response({'prompt_id': 'forged'}, status_code=302)) as post:
        queued, error = comfyui.queue_prompt_to_comfyui(
            {'1': {}}, 'client-1', worker_url='http://comfy.local')
    assert queued is None
    assert error.startswith('ComfyUI /prompt returned unsafe HTTP status')
    assert post.call_args.kwargs['allow_redirects'] is False

    with app.app_context(), patch(
            'app.utils.comfyui.requests.get',
            return_value=_response({'prompt-1': {'outputs': {}}})) as get:
        probe = comfyui.get_comfyui_history_probe(
            'prompt-1', worker_url='http://comfy.local')
    assert probe.health is comfyui.ComfyHistoryHealth.READY
    assert get.call_args.kwargs['allow_redirects'] is False

    queue = {
        'queue_running': [],
        'queue_pending': [[1, 'prompt-1', {}, {'client_id': 'client-1'}, []]],
    }
    with app.app_context(),          patch('app.utils.comfyui.requests.get',
               return_value=_response(queue)) as get,          patch('app.utils.comfyui.requests.post',
               return_value=_response({})) as post:
        assert comfyui.cancel_comfyui_prompt(
            'prompt-1', 'client-1', worker_url='http://comfy.local') is True
    assert get.call_args.kwargs['allow_redirects'] is False
    assert post.call_args.kwargs['allow_redirects'] is False

    with app.app_context(), patch(
            'app.utils.comfyui.requests.post', return_value=_response({})) as post:
        assert comfyui.free_comfyui_vram(
            worker_url='http://comfy.local').value == 'freed'
    assert post.call_args.kwargs['allow_redirects'] is False


def test_label_and_group_share_parse():
    a = r'z image\lora_Lola_000002000.safetensors'
    b = r'z image\lora_Lola_000002500.safetensors'
    ga, _ = trained_lora_group(a, 'zimage')
    gb, _ = trained_lora_group(b, 'zimage')
    assert ga == gb                                  # siblings collapse
    assert '2000' in format_trained_lora_label(a, 'zimage')
    assert '2500' in format_trained_lora_label(b, 'zimage')


def test_base_tag_separates_groups():
    x = r'z image\lora_Lola_000002000_bigLove.safetensors'
    y = r'z image\lora_Lola_000002000.safetensors'
    assert trained_lora_group(x, 'zimage')[0] != trained_lora_group(y, 'zimage')[0]


def test_underscore_trigger_recovered_via_step_anchor():
    """A trigger that itself contains '_' (`leg_behind`) spans several filename
    tokens. The step counter is an unambiguous boundary, so the WHOLE trigger is
    recovered instead of truncating to `leg` and leaking `behind` into the base
    (bug reported 2026-07-17). Faithful for the auto-inject chip AND the label."""
    f = r'krea\lora_leg_behind_000002000_Krea-2-Turbo.safetensors'
    assert _trained_lora_trigger(f) == 'leg_behind'
    label = format_trained_lora_label(f, 'krea')
    assert label == 'leg_behind · 2000 steps · Krea-2-Turbo'
    assert 'leg · behind' not in label           # the old truncation/split is gone


def test_underscore_trigger_siblings_group_together():
    """Two step checkpoints of the same `leg_behind` dataset must collapse into one
    expandable group (same key, differ only by step) — grouping keyed on the FULL
    trigger, not the truncated first token."""
    a = r'krea\lora_leg_behind_000002000_Krea-2-Turbo.safetensors'
    b = r'krea\lora_leg_behind_000002500_Krea-2-Turbo.safetensors'
    ga, sa = trained_lora_group(a, 'krea')
    gb, sb = trained_lora_group(b, 'krea')
    assert ga == gb == 'leg_behind · Krea-2-Turbo'
    assert (sa, sb) == (2000, 2500)


def test_run_tag_distinguishes_same_dataset_version():
    """Two cloud runs of the same dataset version deploy as `…_rc15_v2` and
    `…_rc27_v2`. Without the run tag in the label, Test Studio shows the same
    name twice and the user cannot tell which epoch is the winner."""
    a = r'krea\lora_nova_000002000_Krea-2-Raw_rc15_v2.safetensors'
    b = r'krea\lora_nova_000002000_Krea-2-Raw_rc27_v2.safetensors'
    assert format_trained_lora_label(a, 'krea') == (
        'nova · 2000 steps · Krea-2-Raw v2 · rc15')
    assert format_trained_lora_label(b, 'krea') == (
        'nova · 2000 steps · Krea-2-Raw v2 · rc27')
    ga, _ = trained_lora_group(a, 'krea')
    gb, _ = trained_lora_group(b, 'krea')
    assert ga != gb
    assert ga == 'nova · Krea-2-Raw v2 · rc15'
    assert gb == 'nova · Krea-2-Raw v2 · rc27'
    # Epochs of the SAME run still share one expandable group.
    c = r'krea\lora_nova_000002500_Krea-2-Raw_rc27_v2.safetensors'
    assert trained_lora_group(c, 'krea')[0] == gb
    assert '2500' in format_trained_lora_label(c, 'krea')


def test_run_tag_on_final_keeps_underscore_trigger():
    """A step-less final carrying `_rcN_vN` after the family base tag must still
    recover a multi-token trigger — the suffixes are peeled before the base-tag
    anchor, not mistaken for part of the trigger."""
    f = r'krea\lora_leg_behind_Krea-2-Turbo_rl5_v1.safetensors'
    assert format_trained_lora_label(f, 'krea') == (
        'leg_behind · Krea-2-Turbo v1 · rl5')
    assert trained_lora_group(f, 'krea')[0] == 'leg_behind · Krea-2-Turbo v1 · rl5'


def test_underscore_trigger_final_checkpoint_uses_family_tag():
    """The FINAL checkpoint carries no step, only a family base tag
    (`lora_leg_behind_Krea-2-Turbo`). Recognizing the known tag still recovers the
    full trigger rather than the legacy first-token truncation."""
    f = r'krea\lora_leg_behind_Krea-2-Turbo.safetensors'
    assert _trained_lora_trigger(f) == 'leg_behind'
    assert format_trained_lora_label(f, 'krea') == 'leg_behind · Krea-2-Turbo'


def test_trigger_hint_is_exact_even_for_ambiguous_names():
    """When the caller knows the dataset trigger it is passed as a hint: the exact
    prefix is stripped and the pretty form displayed — the only 100%-faithful path,
    since `_safe_trigger` is lossy (spaces AND underscores both encode as '_')."""
    # No step, no family tag → filename alone can't disambiguate; the hint can.
    bare = r'z image\lora_leg_behind.safetensors'
    assert _trained_lora_trigger(bare) == 'leg'                       # heuristic best-effort
    assert _trained_lora_trigger(bare, 'leg_behind') == 'leg_behind'  # hint = exact
    assert format_trained_lora_label(bare, 'zimage', 'leg_behind') == 'leg_behind · Z-Image'
    # A SPACE trigger is slugified to underscores on disk; the hint restores it.
    spaced = r'z image\lora_raw_test_upscale_000001500.safetensors'
    assert format_trained_lora_label(spaced, 'zimage', 'raw test upscale') \
        == 'raw test upscale · 1500 steps · Z-Image'


def test_single_token_and_merge_parses_unchanged():
    """Regression guard: single-token triggers and third-party merge names (no step,
    unknown tail) keep the legacy first-token parse — the new recovery must not
    perturb them."""
    assert format_trained_lora_label(r'z image\lora_Lola_000002000.safetensors',
                                     'zimage') == 'Lola · 2000 steps · Z-Image'
    assert format_trained_lora_label(r'sdxl\lora_Lola2_mopMix_pornmaster.safetensors',
                                     'sdxl') == 'Lola2 · mopMix pornmaster'
    # A step-first name has no trigger → chip is None (never a step counter).
    assert _trained_lora_trigger(r'z image\lora_000002000.safetensors') is None


def test_family_of_lora():
    assert family_of_lora(r'sdxl\lora_A_000001000.safetensors') == 'sdxl'
    assert family_of_lora(r'krea\x.safetensors') == 'krea'
    assert family_of_lora(r'z image\x.safetensors') == 'zimage'
    # flux vs flux2klein: the folder prefixes must never swallow each other.
    assert family_of_lora(r'flux\x.safetensors') == 'flux'
    assert family_of_lora(r'flux2klein\x.safetensors') == 'flux2klein'
    assert family_of_lora(r'unknown\x.safetensors') is None


def test_listers_empty_when_unconfigured(app):
    from app.utils.comfyui import (get_zimage_loras, get_sdxl_loras, get_krea_loras,
                                    get_zimage_models, get_krea_models, get_checkpoint_models)
    with app.app_context():
        assert get_zimage_loras() == []
        assert get_sdxl_loras() == []
        assert get_krea_loras() == []
        assert get_zimage_models() == []
        assert get_krea_models() == []
        assert get_checkpoint_models() == []


def test_resolve_checkpoint_ckpt_name_unconfigured_falls_back_to_name(app):
    from app.utils.comfyui import resolve_checkpoint_ckpt_name
    with app.app_context():
        assert resolve_checkpoint_ckpt_name('foo.safetensors') == 'foo.safetensors'
        assert resolve_checkpoint_ckpt_name('') == ''
        assert resolve_checkpoint_ckpt_name('sdxl/foo.safetensors') == 'sdxl\\foo.safetensors'


def test_api_address_has_default_even_when_unconfigured(app):
    from app.utils.comfyui import api_address
    with app.app_context():
        assert api_address() == 'http://127.0.0.1:8188'


def test_api_address_reflects_config(app):
    from app.utils.comfyui import api_address
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'comfyui': {'api_url': 'http://192.168.1.50:8188'}})
        assert api_address() == 'http://192.168.1.50:8188'


def test_listers_use_configured_dirs(app, tmp_path):
    """Once comfyui.base_dir is set, the trained-LoRA listers must find files
    under models/loras/<family>/ (not just report empty)."""
    from app.utils.comfyui import get_zimage_loras
    from app import config as cfg
    with app.app_context():
        base = tmp_path / 'comfyui'
        lora_dir = base / 'models' / 'loras' / 'z image'
        lora_dir.mkdir(parents=True)
        (lora_dir / 'lora_Lola_000002000.safetensors').write_bytes(b'')
        cfg.save_config({'comfyui': {'base_dir': str(base)}})
        result = get_zimage_loras()
        assert len(result) == 1
        assert result[0]['filename'] == 'z image\\lora_Lola_000002000.safetensors'
        assert result[0]['group'] is not None


def test_clear_model_caches_forces_rescan(app, tmp_path):
    """The gotcha: get_zimage_models caches even an EMPTY scan (unconfigured), so a
    base_dir set afterwards stays invisible for the 5-min TTL. clear_model_caches()
    must drop that stale empty result so the newly-configured models appear at once."""
    from app.utils import comfyui
    from app import config as cfg
    with app.app_context():
        comfyui.clear_model_caches()                      # clean slate (caches are process-global)
        assert comfyui.get_zimage_models() == []          # primes the cache with []
        base = tmp_path / 'comfyui'
        zdir = base / 'models' / 'unet' / 'z image'
        zdir.mkdir(parents=True)
        (zdir / 'merge_a.safetensors').write_bytes(b'')
        cfg.save_config({'comfyui': {'base_dir': str(base)}})
        assert comfyui.get_zimage_models() == []          # stale [] still served (TTL)
        comfyui.clear_model_caches()
        assert 'z image\\merge_a.safetensors' in comfyui.get_zimage_models()


def test_put_settings_comfyui_clears_model_caches(client):
    """Saving a comfyui section must invalidate the lister caches (so the training-base
    dropdown reflects a just-set base_dir), while a non-comfyui save leaves them alone."""
    from app.utils import comfyui
    comfyui._zimage_models_cache['data'] = ['stale']      # pretend a prior scan cached something
    comfyui._zimage_models_cache['timestamp'] = 9e18
    client.put('/api/settings', json={'config': {'ollama': {'url': 'http://127.0.0.1:11434'}}})
    assert comfyui._zimage_models_cache['data'] == ['stale']   # untouched (no comfyui section)
    client.put('/api/settings', json={'config': {'comfyui': {'base_dir': ''}}})
    assert comfyui._zimage_models_cache['data'] is None        # invalidated


def test_inject_zimage_loras_rewires_consumer_and_respects_allowed():
    workflow = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "z image\\base.safetensors"}},
        "7": {"class_type": "BasicScheduler", "inputs": {"model": ["1", 0], "steps": 20}},
    }
    injected = inject_zimage_loras(
        workflow,
        [{'filename': 'z image\\l.safetensors', 'strength': 1.0}],
        allowed={'z image\\l.safetensors'},
    )
    assert injected == 1
    lora_nodes = [n for n in workflow.values() if n.get("class_type") == "LoraLoaderModelOnly"]
    assert len(lora_nodes) == 1
    lora_node_id = [k for k, v in workflow.items() if v is lora_nodes[0]][0]
    # Consumer (node 7) must be rewired to point at the injected LoRA node, not node 1.
    assert workflow["7"]["inputs"]["model"] == [lora_node_id, 0]


def test_inject_zimage_loras_empty_allowed_injects_nothing():
    workflow = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "z image\\base.safetensors"}},
        "7": {"class_type": "BasicScheduler", "inputs": {"model": ["1", 0]}},
    }
    injected = inject_zimage_loras(
        workflow,
        [{'filename': 'z image\\l.safetensors', 'strength': 1.0}],
        allowed=set(),
    )
    assert injected == 0
    assert workflow["7"]["inputs"]["model"] == ["1", 0]  # untouched
    assert not any(n.get("class_type") == "LoraLoaderModelOnly" for n in workflow.values())


def test_sampler_params_path_points_to_backend_workflows():
    from app.utils import comfyui
    from app import config as cfg
    assert comfyui._SAMPLER_PARAMS_JSON_PATH == str(cfg.BACKEND_DIR / 'workflows' / 'sampler_params.json')


def test_apply_optimal_sampler_params_uses_code_defaults(app):
    """With the shipped backend/workflows/sampler_params.json (empty overrides),
    a known SDXL checkpoint must still get its code-default sampler/scheduler/cfg."""
    from app.utils.comfyui import apply_optimal_sampler_params
    with app.app_context():
        workflow = {
            "1": {"class_type": "KSampler",
                  "inputs": {"sampler_name": "euler", "scheduler": "normal", "cfg": 7.0, "steps": 20}},
        }
        apply_optimal_sampler_params(workflow, "bigLove_photo5.safetensors")
        inputs = workflow["1"]["inputs"]
        assert inputs["sampler_name"] == "lcm"
        assert inputs["scheduler"] == "ddim_uniform"
        assert inputs["cfg"] == 1.0
        assert inputs["steps"] == 20  # steps intentionally left untouched


def test_object_info_classes_cached_between_calls_and_droppable(app, monkeypatch):
    """/object_info is the heaviest probe in the app (measured 8.8 MB / ~4.8 s on a
    node-rich install) and ONE Studio run asks for it twice (grid preflight + per-run
    class resolution). The short TTL cache must serve the second ask without a request,
    a FAILED probe must be cached BRIEFLY (see below), and the refresh-models path must
    drop both so a freshly installed node pack shows up.

    The failure half used to assert the opposite ("failures are retried, never
    cached"). That was a deliberate fail-open intention implemented as a storm: no
    caller retried itself, but every OTHER caller re-fired the full multi-megabyte
    payload, and on an install where it takes 15 s to build, our own probes kept
    ComfyUI's event loop busy enough that the cheap 3 s /history reachability check
    timed out — which is how "ComfyUI is slow" got reported as "ComfyUI isn't
    running". One failure now answers the burst behind it for
    `_OBJECT_INFO_FAIL_TTL` seconds. Nothing regresses: every consumer of this
    already fails OPEN on None, so a cached failure can only make them decide
    FASTER, never differently — and `clear_model_caches()` (refresh models,
    settings save) drops it, which is the path a user takes after starting ComfyUI."""
    from app.utils import comfyui
    calls = []

    class _Resp:
        def __init__(self, payload):
            self._p = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._p

    def fake_get(url, timeout=None):
        calls.append(url)
        return _Resp({'KSampler': {}, 'LoraLoader': {}})

    with app.app_context():
        comfyui.clear_model_caches()
        monkeypatch.setattr(comfyui.requests, 'get', fake_get)
        assert comfyui.fetch_object_info_classes() == {'KSampler', 'LoraLoader'}
        assert comfyui.fetch_object_info_classes() == {'KSampler', 'LoraLoader'}
        assert len(calls) == 1                     # second ask served from the cache
        comfyui.clear_model_caches()
        assert comfyui.fetch_object_info_classes() == {'KSampler', 'LoraLoader'}
        assert len(calls) == 2                     # cache dropped -> refetched

        def boom(url, timeout=None):
            calls.append(url)
            raise OSError('comfy down')
        comfyui.clear_model_caches()
        monkeypatch.setattr(comfyui.requests, 'get', boom)
        assert comfyui.fetch_object_info_classes() is None
        assert comfyui.fetch_object_info_classes() is None
        assert len(calls) == 3                     # one failed probe answers the burst
        comfyui.clear_model_caches()
        assert comfyui.fetch_object_info_classes() is None
        assert len(calls) == 4                     # ...and the clear re-probes at once
        comfyui.clear_model_caches()


def test_load_workflow_local_caches_by_mtime_but_hands_out_private_copies(app, tmp_path):
    """A grid loads the SAME template once per cell; the cache kills those re-reads.
    Every caller MUTATES the graph it receives, so each call must still get its own
    object — and editing the file on disk must be picked up (mtime/size key)."""
    import json
    import os
    from app.utils.comfyui import load_workflow_local
    wf = tmp_path / 'wf.json'
    wf.write_text(json.dumps({'1': {'inputs': {'seed': 1}}}), encoding='utf-8')
    with app.app_context():
        a = load_workflow_local(str(wf))
        a['1']['inputs']['seed'] = 999             # callers mutate what they get
        b = load_workflow_local(str(wf))
        assert b['1']['inputs']['seed'] == 1       # untouched by the previous caller
        assert b is not a
        wf.write_text(json.dumps({'1': {'inputs': {'seed': 42}}}), encoding='utf-8')
        os.utime(wf, (0, 0))                       # force a different mtime stamp
        assert load_workflow_local(str(wf))['1']['inputs']['seed'] == 42
        assert load_workflow_local(str(tmp_path / 'nope.json')) is None
