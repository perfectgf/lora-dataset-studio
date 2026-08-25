"""The shipped workflows must run on a VANILLA ComfyUI.

WHY THIS FILE EXISTS (2026-07-27)
---------------------------------
Reported by IndependentProcess0 (Reddit): generating variations died on his
install with a ComfyUI console error he had to go and find himself —

    Value not in list: scheduler: 'beta57' not in ['simple', 'sgm_uniform',
    'karras', 'exponential', 'ddim_uniform', 'beta', 'normal',
    'linear_quadratic', 'kl_optimal']

— while plain Klein generation in ComfyUI worked fine for him.

His ComfyUI was not old. It was NORMAL. The list he pasted is character-for-
character the core scheduler list (`comfy/samplers.py`, SCHEDULER_HANDLERS).
`beta57` has never been part of ComfyUI: it is added by the third-party node pack
RES4LYF, which monkey-patches the CORE enumeration at import time —

    RES4LYF/__init__.py
        SCHEDULER_HANDLERS["beta57"] = SchedulerHandler(
            handler=partial(comfy.samplers.beta_scheduler, alpha=0.5, beta=0.7))
        SCHEDULER_NAMES.append("beta57")

— so on a machine carrying that pack, even a plain core KSampler accepts
`beta57`. `improve skin.json` and `klein_inpaint.json` were captured on such a
machine and the value was frozen into them. Every user WITHOUT RES4LYF — most of
them — could not generate at all, and nothing in the graph hinted at a dependency:
not one third-party node appears in it.

THAT is the trap this file exists to catch. A node pack that extends a standard
enumeration makes a graph non-portable with NO visible third-party node, so the
usual "are all the node classes present?" preflight passes and the user finds out
at their first generation.

Both files now use `scheduler: "simple"`, which every ComfyUI has. It was picked
by alignment, not taste: four other shipped graphs already use `simple`
(ZImage_bigLove_ZT3_optimal, krea2_turbo, krea2_turbo_img2img, and the Krea graph
built in krea_edit_helper), and both Klein graphs already sampled with `euler`.
This DOES change how Klein images render compared to the old `beta57` — a
deliberate, one-time, everyone-at-once change, which is the point: one render
path for every install, not a per-user approximation.

WHY A STATIC TEST AND NOT ONLY THE RUNTIME CHECK
------------------------------------------------
There is also a runtime check (utils.comfyui.unsupported_enum_values) that turns
ComfyUI's raw 400 into a readable sentence. It is a good net but it is the LAST
one: it only fires on a user's machine, after shipping. This test fires on the
author's machine, offline, with no ComfyUI running — at the moment someone pastes
a workflow exported from an equipped install, which is exactly how the bug got in.
Cheap, deterministic, and it names the fix in its own failure message.

The reference lists below are transcribed from ComfyUI v0.28.3 (`comfy/samplers.py`
SCHEDULER_HANDLERS / SAMPLER_NAMES). They are the FLOOR, not the ceiling: they are
deliberately the values available on a plain install, so a value that is core but
very recent still gets a human look before it ships.
"""
import json
from pathlib import Path

import pytest

WORKFLOW_DIR = Path(__file__).resolve().parents[1] / 'workflows'

# comfy/samplers.py :: SCHEDULER_HANDLERS — the exact list the reporter's ComfyUI
# published, and unchanged across every release checked.
CORE_SCHEDULERS = frozenset({
    'simple', 'sgm_uniform', 'karras', 'exponential', 'ddim_uniform', 'beta',
    'normal', 'linear_quadratic', 'kl_optimal',
})

# comfy/samplers.py :: SAMPLER_NAMES = KSAMPLER_NAMES + ["ddim", "uni_pc", "uni_pc_bh2"]
CORE_SAMPLERS = frozenset({
    'euler', 'euler_cfg_pp', 'euler_ancestral', 'euler_ancestral_cfg_pp', 'heun',
    'heunpp2', 'exp_heun_2_x0', 'exp_heun_2_x0_sde', 'dpm_2', 'dpm_2_ancestral',
    'lms', 'dpm_fast', 'dpm_adaptive', 'dpmpp_2s_ancestral',
    'dpmpp_2s_ancestral_cfg_pp', 'dpmpp_sde', 'dpmpp_sde_gpu', 'dpmpp_2m',
    'dpmpp_2m_cfg_pp', 'dpmpp_2m_sde', 'dpmpp_2m_sde_gpu', 'dpmpp_2m_sde_heun',
    'dpmpp_2m_sde_heun_gpu', 'dpmpp_3m_sde', 'dpmpp_3m_sde_gpu', 'ddpm', 'lcm',
    'ipndm', 'ipndm_v', 'deis', 'res_multistep', 'res_multistep_cfg_pp',
    'res_multistep_ancestral', 'res_multistep_ancestral_cfg_pp',
    'gradient_estimation', 'gradient_estimation_cfg_pp', 'er_sde', 'seeds_2',
    'seeds_3', 'sa_solver', 'sa_solver_pece', 'ddim', 'uni_pc', 'uni_pc_bh2',
})

# nodes.py :: CLIPLoader `type`.
CORE_CLIP_TYPES = frozenset({
    'stable_diffusion', 'stable_cascade', 'sd3', 'stable_audio', 'mochi', 'ltxv',
    'pixart', 'cosmos', 'lumina2', 'wan', 'hidream', 'chroma', 'ace', 'omnigen2',
    'qwen_image', 'hunyuan_image', 'flux2', 'ovis', 'longcat_image', 'cogvideox',
    'lens', 'pixeldit', 'ideogram4', 'boogu', 'krea2',
})

# UNETLoader / CLIPLoader `weight_dtype` and `device`.
CORE_WEIGHT_DTYPES = frozenset({'default', 'fp8_e4m3fn', 'fp8_e4m3fn_fast', 'fp8_e5m2'})
CORE_DEVICES = frozenset({'default', 'cpu'})

CORE_ENUMS = {
    'scheduler': CORE_SCHEDULERS,
    'sampler_name': CORE_SAMPLERS,
    'type': CORE_CLIP_TYPES,
    'weight_dtype': CORE_WEIGHT_DTYPES,
    'device': CORE_DEVICES,
}

# Node classes our graphs use that a vanilla ComfyUI does NOT provide. Every entry
# is a DECLARED dependency: it must be gated by a preflight that names the pack,
# because the graph cannot run without it. Adding a node to a shipped workflow
# without listing it here fails the inventory test below — which is the point: it
# forces the question "does a stock install have this?" to be answered once, in
# the repo, instead of being discovered by a user.
DECLARED_THIRD_PARTY_NODES = {
    # Test Studio's HQ graph. Gated by lora_test_studio's node preflight.
    'DetailDaemonSamplerNode': 'ComfyUI-Detail-Daemon',
    # Krea 2 Edit. Gated by krea_edit_helper.krea_missing_nodes(), which the Setup
    # screen turns into an "install the pack" action.
    'ConditioningKrea2Rebalance': 'comfyui-krea2edit',
    'Krea2EditModelPatch': 'comfyui-krea2edit',
    'Krea2EditGroundedEncode': 'comfyui-krea2edit',
    # Masked Repair's sampler. Gated by lanpaint_helper.lanpaint_missing_nodes(),
    # checked by _run_klein_mask_job before it queues anything, and installable
    # from Setup's "LanPaint sampler" row.
    'LanPaint_KSampler': 'LanPaint',
}

# The Klein lane is the one that broke, and the one the app leans on hardest
# (variations, watermark cleaning, rescue). It must need NOTHING but a stock
# ComfyUI — no node pack, and no enum value borrowed from one.
VANILLA_ONLY_WORKFLOWS = ('improve skin.json', 'klein_inpaint.json')


def _graphs():
    """(filename, graph) for every shipped API-format workflow. sampler_params.json
    is an override store, not a graph, so it has no nodes and is skipped."""
    for path in sorted(WORKFLOW_DIR.glob('*.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(data, dict) and any(
                isinstance(v, dict) and v.get('class_type') for v in data.values()):
            yield path.name, data


def _nodes(graph):
    for node_id, node in graph.items():
        if isinstance(node, dict) and node.get('class_type'):
            yield node_id, node


def test_there_is_something_to_check():
    """A glob that silently matches nothing would make every assertion below pass
    for the wrong reason."""
    assert len(list(_graphs())) >= 6


@pytest.mark.parametrize('name,graph', list(_graphs()), ids=lambda v: v if isinstance(v, str) else '')
def test_every_pinned_enum_value_exists_in_a_vanilla_comfyui(name, graph):
    """The regression guard for the reported bug. `beta57` fails here loudly, with
    the reason, instead of reaching a user as ComfyUI's raw 400."""
    offenders = []
    for node_id, node in _nodes(graph):
        for field, value in (node.get('inputs') or {}).items():
            allowed = CORE_ENUMS.get(field)
            if allowed is None or not isinstance(value, str) or value in allowed:
                continue
            offenders.append(f'{name} node {node_id} ({node["class_type"]}): '
                             f'{field}="{value}"')
    assert not offenders, (
        'These values do not exist in a stock ComfyUI, so nobody without the '
        'right third-party node pack can run this graph:\n  '
        + '\n  '.join(offenders)
        + '\n\nA pack can inject values into a CORE enumeration (RES4LYF does this '
          'for the "beta57" scheduler), so a graph can be non-portable while '
          'containing no third-party node at all. Pick a value from the core list '
          'in this file, or declare the dependency and gate it in a preflight.')


@pytest.mark.parametrize('name,graph', list(_graphs()), ids=lambda v: v if isinstance(v, str) else '')
def test_third_party_node_classes_are_declared(name, graph):
    """Not a ban — an inventory. Two of our engines legitimately need a pack; what
    must never happen is a pack dependency nobody wrote down and nobody gated."""
    undeclared = sorted({
        node['class_type'] for _, node in _nodes(graph)
        if node['class_type'] not in VANILLA_NODE_ALLOWLIST
        and node['class_type'] not in DECLARED_THIRD_PARTY_NODES})
    assert not undeclared, (
        f'{name} uses node class(es) not known to be part of a stock ComfyUI: '
        f'{undeclared}. If they come from a pack, add them to '
        'DECLARED_THIRD_PARTY_NODES and make sure a preflight names that pack; if '
        'they are core, add them to VANILLA_NODE_ALLOWLIST.')


def test_the_klein_lane_needs_no_third_party_pack_at_all():
    """Klein is the default local engine and the one this bug took out. Its graphs
    must stay runnable on a ComfyUI with zero custom nodes — including their
    widget VALUES, which is the part that was missed."""
    for name in VANILLA_ONLY_WORKFLOWS:
        graph = json.loads((WORKFLOW_DIR / name).read_text(encoding='utf-8'))
        for node_id, node in _nodes(graph):
            assert node['class_type'] not in DECLARED_THIRD_PARTY_NODES, (
                f'{name} node {node_id} needs a third-party pack')
            for field, value in (node.get('inputs') or {}).items():
                allowed = CORE_ENUMS.get(field)
                if allowed is not None and isinstance(value, str):
                    assert value in allowed, f'{name} node {node_id}: {field}="{value}"'


def test_the_klein_graphs_sample_the_way_the_rest_of_the_app_does():
    """Pins the replacement itself. `simple` was not a taste call: four other
    shipped graphs already use it, and both Klein graphs already sample with
    `euler` — the canonical ComfyUI pairing for this model family. If someone
    "improves" this back to a value that only exists on their own machine, the
    test above catches it; this one catches a silent drift of the render."""
    for name in VANILLA_ONLY_WORKFLOWS:
        graph = json.loads((WORKFLOW_DIR / name).read_text(encoding='utf-8'))
        samplers = {(n['inputs'].get('sampler_name'), n['inputs'].get('scheduler'))
                    for _, n in _nodes(graph) if n['class_type'] == 'KSampler'}
        assert samplers == {('euler', 'simple')}, f'{name}: {samplers}'


# Core node classes the shipped graphs use, verified against the ComfyUI v0.28.3
# source (nodes.py + comfy_extras/*, both the legacy NODE_CLASS_MAPPINGS and the
# V3 `node_id=` schema).
VANILLA_NODE_ALLOWLIST = frozenset({
    'BasicGuider', 'BasicScheduler', 'CFGGuider', 'CLIPLoader', 'CLIPTextEncode',
    'CheckpointLoaderSimple', 'ConditioningZeroOut', 'EmptyFlux2LatentImage',
    'EmptyLatentImage', 'EmptySD3LatentImage', 'GetImageSize',
    'ImageScaleToTotalPixels', 'InpaintModelConditioning', 'KSampler',
    'KSamplerSelect', 'LatentUpscaleBy', 'LoadImage', 'LoadImageMask',
    'LoraLoader',
    'LoraLoaderModelOnly', 'ModelSamplingFlux', 'PatchModelAddDownscale',
    'PreviewImage', 'PrimitiveInt', 'RandomNoise', 'ReferenceLatent',
    'SamplerCustomAdvanced', 'SaveImage', 'SetLatentNoiseMask', 'UNETLoader',
    'VAEDecode', 'VAEEncode', 'VAELoader',
    # 📷 The camera-angle graph's five. All core, and CHECKED rather than
    # assumed — a live ComfyUI 0.30.1 was asked for each one's
    # `python_module` and every answer was a `comfy_extras.*` module, i.e.
    # shipped with ComfyUI itself and not a pack:
    #   CFGNorm                               comfy_extras.nodes_cfg
    #   FluxKontextImageScale                 comfy_extras.nodes_flux
    #   FluxKontextMultiReferenceLatentMethod comfy_extras.nodes_flux
    #   ModelSamplingAuraFlow                 comfy_extras.nodes_model_advanced
    #   TextEncodeQwenImageEditPlus           comfy_extras.nodes_qwen
    # They are recent, so a much older ComfyUI will not have them — that is a
    # /object_info gap the preflight reports, not a missing pack to install.
    'CFGNorm', 'FluxKontextImageScale', 'FluxKontextMultiReferenceLatentMethod',
    'ModelSamplingAuraFlow', 'TextEncodeQwenImageEditPlus',
})
