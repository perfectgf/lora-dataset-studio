"""Sparse attention keeps working when H3-Optimizations moves.

LDS's sparse graph was written for H3-Optimizations 0.2.14. From 0.2.31 the
pack renamed the backend it asks for ("Kitchen INT8") and made three new
controls required, so ComfyUI refused every sparse render ("Value not in list",
"Required input is missing"). The new controls also default to a different
clip (an Early KV ramp, 1x8x8 token tiles, calibrated per-head budgets).

The 0.4.0 schema below is copied from that release's own source
(h3_optimizations/nodes.py, plan.py, memory_migration_node.py, commit HEAD of
2026-11-24): names, options and defaults as the node declares them.
"""
import pytest

from app.services import video_test_studio as legacy
from app.utils import comfyui
from lds_sdk import h3_reference_catalog as vrc
from lds_sdk import h3_render


def combo(options, default):
    return ['COMBO', {'options': list(options), 'default': default}]


SPARSE_0_2_14 = {'input': {'required': {
    'model': ['MODEL'],
    'video_budget': ['FLOAT', {'default': 0.3}], 'early_steps': ['INT', {'default': 2}],
    'early_kv': ['FLOAT', {'default': 0.5}], 'late_steps': ['INT', {'default': 2}],
    'late_kv': ['FLOAT', {'default': 0.5}],
    'backend': [['Kitchen INT8', 'Sparse Sage'], {'default': 'Kitchen INT8'}],
}}}

BACKENDS_0_4_0 = ('Kitchen INT8 64Q x 32KV (Quality)', 'Kitchen INT8 64Q x 64KV (Quality)',
                  'Kitchen INT8 64Q x 128KV (Faster)', 'FROST BF16 64Q x 64KV (SM89)',
                  'Sparse Sage (GPU-specific geometry)', 'BF16 Triton 64Q x 64KV',
                  'FP8 FlexAttention 64Q x 64KV')
SPARSE_0_4_0 = {'input': {'required': {
    'model': ['MODEL', {}],
    'video_budget': ['FLOAT', {'default': 0.15}], 'early_steps': ['INT', {'default': 8}],
    'early_kv': ['FLOAT', {'default': 41.0 / 60.0}], 'late_steps': ['INT', {'default': 0}],
    'late_kv': ['FLOAT', {'default': 0.5}],
    'backend': combo(BACKENDS_0_4_0, 'Kitchen INT8 64Q x 32KV (Quality)'),
    'early_schedule': combo(('Hold', 'Ramp'), 'Ramp'),
    'video_token_order': combo(('1x8x8', '1x16x4', '4x4x4', 'Raster (stock H3 order)'), '1x8x8'),
    'head_budgets': combo(('Calibrated', 'Off'), 'Calibrated'),
}}}

MEMORY_0_4_0 = {'input': {'required': {
    'model': ['MODEL', {}],
    'fused_qkv': combo(('auto', 'off'), 'auto'), 'mlp_memory': combo(('auto', 'off'), 'auto'),
    'chunk_rows': ['INT', {'default': 4096}], 'preserve_precision': ['BOOLEAN', {'default': True}],
    'precision_mode': combo(('Auto', 'BF16', 'Preserve native', 'Force quant'), 'Auto'),
    'qkv_streaming_mode': combo(('Off', 'Auto', 'Forced'), 'Auto'),
    'embedding_memory_mode': combo(('Auto', 'Stock', 'Release early'), 'Auto'),
    'kitchen_v_memory_mode': combo(('Standard', 'Lower VRAM (slower)'), 'Standard'),
}}}


def _sparse_node(module, monkeypatch, infos, **build):
    monkeypatch.setattr(comfyui, 'fetch_node_info', lambda cls, **k: infos.get(cls))
    wf = module.build_workflow(prompt='a quiet street', mode='t2v', sparse='default', **build)['workflow']
    return wf, wf[module.N_SPARSE]['inputs']


@pytest.mark.parametrize('module', [h3_render, legacy], ids=['sdk', 'legacy-core'])
def test_the_release_lds_was_tuned_on_gets_the_graph_unchanged(module, monkeypatch):
    _, with_old = _sparse_node(module, monkeypatch, {'H3SparseAttentionAdvanced': SPARSE_0_2_14})
    _, unprobed = _sparse_node(module, monkeypatch, {})
    assert with_old == unprobed
    assert with_old['backend'] == 'Kitchen INT8'
    assert not {'early_schedule', 'video_token_order', 'head_budgets'} & with_old.keys()


@pytest.mark.parametrize('module', [h3_render, legacy], ids=['sdk', 'legacy-core'])
def test_a_current_release_gets_valid_inputs_that_keep_the_tuned_behaviour(module, monkeypatch):
    _, inputs = _sparse_node(module, monkeypatch, {'H3SparseAttentionAdvanced': SPARSE_0_4_0})
    required = SPARSE_0_4_0['input']['required']
    assert set(required) <= set(inputs), 'every required input is written'
    for name, spec in required.items():
        if spec[0] == 'COMBO':
            assert inputs[name] in spec[1]['options'], (name, inputs[name])
    assert inputs['backend'] == 'Kitchen INT8 64Q x 64KV (Quality)'   # what "Kitchen INT8" was
    assert (inputs['early_schedule'], inputs['video_token_order'], inputs['head_budgets']) == (
        'Hold', 'Raster (stock H3 order)', 'Off')
    # The preset still decides the schedule numbers, never the node's new defaults.
    assert (inputs['early_steps'], inputs['early_kv'], inputs['late_steps']) == (2, 0.5, 2)


def test_the_reference_memory_node_keeps_the_forwards_on_a_current_release(monkeypatch):
    monkeypatch.setattr(comfyui, 'fetch_node_info', lambda cls, **k: {
        'H3SparseAttentionAdvanced': SPARSE_0_4_0, 'H3MemoryOptimization': MEMORY_0_4_0}.get(cls))
    wf = {h3_render.N_GUIDER: {'inputs': {'model': ['6', 0]}},
          '999': {'class_type': vrc.REF_LORA_CLASS, 'inputs': {}}}
    h3_render._graft_sparse(wf, 'default', upscale_armed=False)
    memory = wf[h3_render.N_REF_SPARSE_MEMORY]['inputs']
    assert (memory['fused_qkv'], memory['mlp_memory'], memory['qkv_streaming_mode'],
            memory['precision_mode']) == ('off', 'off', 'Off', 'Preserve native')
    assert (memory['embedding_memory_mode'], memory['kitchen_v_memory_mode']) == ('Stock', 'Standard')
    assert wf[h3_render.N_SPARSE]['inputs']['model'] == [h3_render.N_REF_SPARSE_MEMORY, 0]


def test_both_render_modules_keep_the_same_tuned_values():
    assert legacy.SPARSE_KEEP_TUNED == h3_render.SPARSE_KEEP_TUNED
    assert legacy.SPARSE_RENAMES == h3_render.SPARSE_RENAMES
    assert legacy.SPARSE_BACKEND == h3_render.SPARSE_BACKEND


def test_fit_leaves_inputs_alone_when_the_probe_failed():
    inputs = {'model': ['6', 0], 'backend': 'Kitchen INT8'}
    assert comfyui.fit_node_inputs(None, inputs, prefer={'head_budgets': 'Off'}) == inputs
    assert comfyui.fit_node_inputs({'input': None}, inputs) == inputs


def test_fit_fills_an_unknown_new_required_control_with_the_nodes_default():
    info = {'input': {'required': {
        'model': ['MODEL'], 'backend': [['A', 'B'], {}],
        'new_mode': combo(('x', 'y'), 'y'), 'new_count': ['INT', {'default': 3}],
        'new_link': ['LATENT'],
    }, 'optional': {'extra': combo(('p', 'q'), 'p')}}}
    out = comfyui.fit_node_inputs(info, {'model': ['6', 0], 'backend': 'B'},
                                  prefer={'extra': 'q', 'missing_control': 'z'})
    assert out == {'model': ['6', 0], 'backend': 'B', 'new_mode': 'y', 'new_count': 3, 'extra': 'q'}
    # A choice that disappeared without a listed successor is left for ComfyUI to name.
    assert comfyui.fit_node_inputs(info, {'backend': 'gone'}, renames={'backend': ('C',)})['backend'] == 'gone'
