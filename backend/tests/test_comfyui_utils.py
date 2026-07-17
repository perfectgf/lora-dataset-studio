"""Tests for the lifted app.utils.comfyui module: the shared trained-LoRA
parser (label + group MUST share one parse — the drift-proof invariant),
config-driven listers (empty/None-safe when ComfyUI isn't configured), and
the LoRA-chain injectors (allowed-whitelist respected)."""
from unittest.mock import MagicMock, patch

from app.utils.comfyui import (
    trained_lora_group, format_trained_lora_label, family_of_lora,
    inject_zimage_loras, _trained_lora_trigger,
)


def _response(payload=None):
    response = MagicMock()
    response.json.return_value = payload
    return response


def test_cancel_comfyui_prompt_interrupts_only_matching_running_prompt(app):
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
         patch('app.utils.comfyui.requests.post', return_value=_response({})) as post:
        assert cancel_comfyui_prompt('target-prompt', 'target-job') is True
    post.assert_called_once()
    assert post.call_args.args[0].endswith('/interrupt')


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
