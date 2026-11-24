"""LDS's own MiniMax H3 nodes ship with the core, and SageAttention stays in a
render only when the target ComfyUI can import the `sageattention` package.

Video and Live offered "Install" for the block-attention switch and the
reference LoRA loader while no core carried them: every click failed. And the
graph kept KJNodes' Sage patch whenever its class was registered, although
KJNodes registers it without the package and imports it only when the patch
runs, so every H3 render failed on a ComfyUI with KJNodes and no package.
"""
import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import setup_installer as si
from app.services import video_test_studio as legacy
from app.utils import comfyui
from lds_sdk import h3_reference_catalog, h3_render


def _mapped_classes(folder):
    """NODE_CLASS_MAPPINGS keys, read from the source: the module imports
    ComfyUI and torch, which only a running ComfyUI has."""
    tree = ast.parse(Path(si.cfg.BACKEND_DIR, 'comfy_nodes', folder, '__init__.py').read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', '') == 'NODE_CLASS_MAPPINGS' for t in node.targets):
            return {key.value for key in node.value.keys}
    return set()


@pytest.mark.parametrize('action,folder,cls', [
    (h3_render.BLOCK_ATTN_INSTALL_ACTION, 'lds_h3_attention', h3_render.BLOCK_ATTN_CLASS),
    (h3_reference_catalog.REF_NODE_ACTION, 'lds_h3_reference', h3_reference_catalog.REF_LORA_CLASS),
])
def test_each_offered_h3_node_is_a_real_install_action_that_registers_its_class(action, folder, cls):
    assert action in si.INSTALL_ACTIONS and si._BUNDLED_NODE_PACKS[action]['folder'] == folder
    assert os.path.isdir(si._bundled_pack_source(action))
    assert _mapped_classes(folder) == {cls}


def test_the_switch_carries_its_apache_licence_and_notice():
    folder = Path(si._bundled_pack_source(h3_render.BLOCK_ATTN_INSTALL_ACTION))
    assert 'Apache License' in (folder / 'LICENSE').read_text(encoding='utf-8')
    assert 'jacokon/fasth3-live' in (folder / 'NOTICE').read_text(encoding='utf-8')
    assert 'SPDX-License-Identifier: Apache-2.0' in (folder / 'attention.py').read_text(encoding='utf-8')


def _switch_info(*backends, v3=False):
    combo = ['COMBO', {'options': list(backends)}] if v3 else [list(backends), {'default': backends[0]}]
    return {'input': {'required': {'model': ['MODEL'], 'middle_backend': combo}}}


@pytest.fixture
def portable(tmp_path, monkeypatch):
    """A managed portable layout whose local server is the target."""
    from app.services import comfyui_control as control
    python = tmp_path / 'python_embeded' / 'python.exe'
    site = python.parent / 'Lib' / 'site-packages'
    site.mkdir(parents=True)
    state = SimpleNamespace(site=site, api='http://127.0.0.1:8188')
    monkeypatch.setattr(control, '_validated_portable_layout',
                        lambda: SimpleNamespace(base_dir=tmp_path / 'ComfyUI', python_exe=python))
    monkeypatch.setattr(control.cfg, 'get', lambda key, default=None: state.api if key == 'comfyui.api_url' else default)
    return state


def test_the_probe_reads_the_switch_the_sdk_grafts():
    assert comfyui.ATTENTION_SWITCH_CLASS == h3_render.BLOCK_ATTN_CLASS
    assert comfyui.SAGE_BACKEND == h3_render.ATTN_SAGE


@pytest.mark.parametrize('v3', [False, True])
def test_a_registered_switch_answers_with_comfyuis_own_import(monkeypatch, v3):
    asked = []
    monkeypatch.setattr(comfyui, 'fetch_node_info',
                        lambda cls, worker_url=None: asked.append(cls) or _switch_info(
                            'pytorch attention', 'comfy kitchen attention', 'sage attention', v3=v3))
    classes = {h3_render.BLOCK_ATTN_CLASS, h3_render.SAGE_CLASS}
    assert comfyui.sageattention_importable(classes) is True
    assert asked == [h3_render.BLOCK_ATTN_CLASS]
    monkeypatch.setattr(comfyui, 'fetch_node_info',
                        lambda cls, worker_url=None: _switch_info('pytorch attention', 'comfy kitchen attention', v3=v3))
    assert comfyui.sageattention_importable(classes) is False


def test_a_registered_switch_that_cannot_be_read_is_a_no(monkeypatch, portable):
    """The switch is the better witness: when it is there but unreadable, the
    disk is not consulted instead."""
    (portable.site / 'sageattention').mkdir()
    (portable.site / 'sageattention' / '__init__.py').write_text('', encoding='utf-8')
    monkeypatch.setattr(comfyui, 'fetch_node_info', lambda cls, worker_url=None: None)
    assert comfyui.sageattention_importable({h3_render.BLOCK_ATTN_CLASS}) is False


def test_without_the_switch_the_managed_portable_answers_from_disk(monkeypatch, portable):
    monkeypatch.setattr(comfyui, 'fetch_node_info', lambda *a, **k: pytest.fail('no switch to ask'))
    classes = {h3_render.SAGE_CLASS}
    assert comfyui.sageattention_importable(classes) is False
    (portable.site / 'sageattention').mkdir()
    (portable.site / 'sageattention' / '__init__.py').write_text('', encoding='utf-8')
    assert comfyui.sageattention_importable(classes) is True
    # A remote target or another worker is not this portable's Python.
    assert comfyui.sageattention_importable(classes, worker_url='http://10.0.0.2:8188') is False
    portable.api = 'http://192.168.1.20:8188'
    assert comfyui.sageattention_importable(classes) is False


def test_without_the_switch_and_without_a_managed_portable_nothing_can_tell(monkeypatch):
    from app.services import comfyui_control as control
    monkeypatch.setattr(control, '_validated_portable_layout', lambda: None)
    assert comfyui.sageattention_importable({h3_render.SAGE_CLASS}) is False


@pytest.mark.parametrize('module', [h3_render, legacy], ids=['sdk', 'legacy-core'])
def test_sage_stays_only_with_kjnodes_and_an_importable_package(monkeypatch, module):
    answers = []

    def importable(classes, worker_url=None):
        answers.append(classes)
        return state['importable']

    state = {'importable': True}
    monkeypatch.setattr(comfyui, 'sageattention_importable', importable)
    assert module.sage_available({module.SAGE_CLASS}) is True
    state['importable'] = False
    assert module.sage_available({module.SAGE_CLASS}) is False
    # KJNodes absent, or /object_info unreadable: no question asked, no Sage.
    answers.clear()
    assert module.sage_available({'UNETLoader'}) is False
    assert module.sage_available(set()) is False
    monkeypatch.setattr(module, 'registered_classes', lambda: None)
    assert module.sage_available() is False
    assert answers == []


def test_a_render_without_sage_says_why():
    built = h3_render.build_workflow(prompt='a quiet street', mode='t2v', sage=False)
    assert h3_render.N_SAGE not in built['workflow']
    assert any('sageattention package missing' in note for note in built['notes'])
