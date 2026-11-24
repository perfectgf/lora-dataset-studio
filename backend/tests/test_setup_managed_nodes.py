"""The existing Krea action uses the verified worker on a supported portable."""
from types import SimpleNamespace

import pytest

from app import setup_installer as installer
from app.services import comfyui_control as control
from app.services import comfyui_node_install as nodes


@pytest.fixture
def action(tmp_path, monkeypatch):
    destination = tmp_path / 'custom_nodes/comfyui-krea2edit'
    monkeypatch.setattr(installer, '_node_pack_dest', lambda _: str(destination))
    monkeypatch.setattr(control, '_validated_portable_layout', lambda: SimpleNamespace(base_dir=tmp_path))
    monkeypatch.setattr(installer, '_runs', {'krea_nodes': installer._new_run()})
    monkeypatch.setattr(installer, '_zip_node_pack', lambda *_: pytest.fail('managed preparation must never use the ZIP fallback'))
    return destination


def test_core_preparation_uses_the_immutable_recipe_and_prepared_verdict(action, monkeypatch):
    calls = []
    def plan(recipe, *, owner):
        assert recipe.url == f'https://lds.perfectgf.fr/models/tools/{recipe.sha256}/comfyui-krea2edit.zip'
        assert recipe.sha256 == 'ff1102a1d17f597d8b100afd17bd489a7910325729483398f1385c541f18c171'
        assert recipe.requirements == ()
        calls.append(('plan', owner))
        return {'plan_id': 'reviewed'}
    def prepare(recipe, *, owner, plan_id, log):
        calls.append(('prepare', owner, plan_id))
        return {'state': 'prepared', 'restart_required': True}
    monkeypatch.setattr(nodes, 'plan', plan)
    monkeypatch.setattr(nodes, 'prepare', prepare)
    assert installer._run_node_pack('krea_nodes') == 0
    assert calls == [('plan', 'lds.core'), ('prepare', 'lds.core', 'reviewed')]
    assert 'RESTART ComfyUI when it is idle' in ' '.join(installer._runs['krea_nodes']['log'])


def test_failed_verification_never_falls_back_to_a_mobile_git_branch(action, monkeypatch):
    def fail(*args, **kwargs):
        raise nodes.NodeInstallError('Archive hash mismatch.')
    monkeypatch.setattr(nodes, 'plan', fail)
    assert installer._run_node_pack('krea_nodes') == 1
    assert not action.exists()


def test_an_existing_unmanaged_node_is_preserved(action, monkeypatch):
    action.mkdir(parents=True)
    source = action / '__init__.py'
    source.write_text('# locally managed node\n')
    monkeypatch.setattr(nodes, 'plan', lambda *args, **kwargs: pytest.fail('do not adopt an unmanaged node'))
    assert installer._run_node_pack('krea_nodes') == 0
    assert source.read_text() == '# locally managed node\n'


def test_plugin_node_batch_refuses_an_unsupported_target_before_admission(monkeypatch):
    from app.plugins import environment
    spec = {'plugin': 'sample.nodes', 'node_pack': {'pack': 'Example'}, 'run': lambda log: 0}
    monkeypatch.setattr(installer, 'known_action', lambda _: True)
    monkeypatch.setattr(installer, 'plugin_action_spec', lambda _: spec)
    monkeypatch.setattr(installer, '_plugin_registry', lambda: SimpleNamespace(
        records={'sample.nodes': SimpleNamespace(id='sample.nodes')}, model_downloads={}))
    monkeypatch.setattr(environment, 'check_enabled', lambda *args, **kwargs: None)
    monkeypatch.setattr(environment, 'running', lambda _: False)
    def refuse():
        raise nodes.NodeInstallError('Select a supported ComfyUI installation first.')
    monkeypatch.setattr(nodes, '_target', refuse)
    with pytest.raises(installer.Precondition, match='supported ComfyUI'):
        installer.check_start_preconditions('sample_nodes')


def test_a_dependency_conflict_does_not_start_the_groups_model_downloads(monkeypatch):
    def refuse():
        raise nodes.NodeInstallError('A dependency conflicts with ComfyUI.')
    monkeypatch.setattr(installer, 'install_group_plan', lambda *args: ['sample_nodes', 'sample_model'])
    monkeypatch.setattr(installer, 'plugin_action_spec', lambda action:
                        {'node_preflight': refuse} if action == 'sample_nodes' else None)
    monkeypatch.setattr(installer, 'start', lambda *_: pytest.fail('no model may start after a failed preflight'))
    with pytest.raises(installer.Precondition, match='conflicts with ComfyUI'):
        installer.start_group('sample')
