"""Managed custom nodes: synthetic portable installation, no executable/network/GPU."""
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import stat
import subprocess
from types import SimpleNamespace
import zipfile

from filelock import FileLock
import pytest

from app.services import comfyui_node_install as nodes


def archive_bytes(entries=None):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        for name, body in (entries or {'__init__.py': 'raise AssertionError("never imported by installer")\n',
                                      'install.py': 'raise AssertionError("never executed")\n'}).items():
            archive.writestr(name, body)
    return buffer.getvalue()


def wheel_bytes(name, version, *, files=None, entry_points=None):
    module = name.replace('-', '_')
    info = f'{module}-{version}.dist-info'
    entries = {f'{module}/__init__.py': '# helper',
               f'{info}/METADATA': f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n',
               f'{info}/WHEEL': 'Wheel-Version: 1.0\nRoot-Is-Purelib: true\n'}
    if files is not None:
        entries.pop(f'{module}/__init__.py')
        entries.update(files)
    if entry_points:
        entries[f'{info}/entry_points.txt'] = entry_points
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        for path, value in entries.items():
            archive.writestr(zipfile.ZipInfo(path), value)
    return buffer.getvalue()


def wheel(name='example-helper', version='1.2.3', *, body=None):
    body = body if body is not None else wheel_bytes(name, version)
    return {'metadata': {'name': name, 'version': version}, 'download_info': {
        'url': f'https://lds.perfectgf.fr/models/tools/packages/{name.replace("-", "_")}-{version}-py3-none-any.whl',
        'archive_info': {'hashes': {'sha256': hashlib.sha256(body).hexdigest()}}}}


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    base = tmp_path / 'portable' / 'ComfyUI'
    base.mkdir(parents=True)
    (base / 'main.py').write_text('# synthetic only\n', encoding='utf-8')
    (base / 'models').mkdir()
    python = base.parent / 'python_embeded' / 'python.exe'
    python.parent.mkdir()
    python.write_bytes(b'fake interpreter; never executed')
    (base.parent / 'run_nvidia_gpu.bat').write_text('@echo off\n', encoding='utf-8')
    config = {'comfyui.base_dir': str(base), 'comfyui.api_url': 'http://127.0.0.1:8188'}
    monkeypatch.setattr(nodes.control.cfg, 'is_configured', lambda: True)
    monkeypatch.setattr(nodes.control.cfg, 'get', lambda key, default=None: config.get(key, default))
    body = archive_bytes()
    state = SimpleNamespace(base=base, python=python, config=config, body=body, calls=[],
                            gets=[], installed={'torch': '2.8.0+cu128', 'numpy': '2.2.6'},
                            reports=[wheel()], wheel_bodies={}, fail_install=False, fail_check=False,
                            probe_override={}, after_install=None)
    recipe = nodes.NodeRecipe(id='sample.nodes', version='1.0.0', folder='sample-nodes',
                              url='https://example.org/nodes/v1.zip',
                              sha256=hashlib.sha256(body).hexdigest(), expected_classes=('ExampleNode',))
    state.recipe = recipe

    def run(executable, args):
        assert Path(executable) == state.python, 'must use the ComfyUI interpreter only'
        state.calls.append(list(args))
        if args[:1] == ['-c']:
            return json.dumps({'executable': str(python), 'prefix': str(python.parent),
                               'version': [3, 12, 10], 'abi': 'cpython-312',
                               'install_paths': {'purelib': str(python.parent / 'Lib' / 'site-packages'),
                                                 'platlib': str(python.parent / 'Lib' / 'site-packages'),
                                                 'scripts': str(python.parent / 'Scripts'),
                                                 'data': str(python.parent)},
                               'packages': list(state.installed.items()), **state.probe_override})
        if '--dry-run' in args:
            report = Path(args[args.index('--report') + 1])
            state.constraints = Path(args[args.index('--constraint') + 1]).read_text(encoding='utf-8')
            report.write_text(json.dumps({'version': '1', 'install': state.reports}), encoding='utf-8')
        elif 'install' in args:
            state.locked = Path(args[args.index('-r') + 1]).read_text(encoding='utf-8')
            if state.fail_install:
                state.installed['example-helper'] = '1.2.3'  # pip may have partially succeeded
                raise nodes.NodeInstallError('simulated pip failure')
            state.installed.update({w['metadata']['name']: w['metadata']['version'] for w in state.reports})
            if state.after_install:
                state.after_install()
        elif 'check' in args and state.fail_check:
            raise nodes.NodeInstallError('simulated existing dependency conflict')
        return ''

    class Response:
        status_code = 200

        def __init__(self, body):
            self.body = body
            self.raw = self
            self.consumed = False

        @property
        def headers(self):
            return {'Content-Length': str(len(self.body))}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read1(self, size, decode_content=False):
            if self.consumed:
                return b''
            self.consumed = True
            return self.body

    def get(url, **kwargs):
        state.gets.append((url, kwargs))
        assert kwargs['allow_redirects'] is False
        if url == state.recipe.url:
            return Response(state.body)
        entry = next(w for w in state.reports if w['download_info']['url'] == url)
        name, version = entry['metadata']['name'], entry['metadata']['version']
        return Response(state.wheel_bodies.get(name, wheel_bytes(name, version)))

    monkeypatch.setattr(nodes, '_run', run)
    monkeypatch.setattr(nodes.requests, 'get', get)
    return state


def prepare(state, *, recipe=None, owner='example.one'):
    recipe = recipe or state.recipe
    proposal = nodes.plan(recipe, owner=owner)
    return nodes.prepare(recipe, owner=owner, plan_id=proposal['plan_id'])


def pip_installs(state):
    return [args for args in state.calls if 'install' in args and '--dry-run' not in args]


def test_prepares_files_without_importing_nodes_or_running_install_py(sandbox):
    proposal = nodes.plan(sandbox.recipe, owner='example.one')
    assert proposal['state'] == 'install'
    assert not (sandbox.base / 'custom_nodes').exists()
    assert sandbox.gets == []
    result = nodes.prepare(sandbox.recipe, owner='example.one', plan_id=proposal['plan_id'])
    assert result['state'] == 'prepared' and result['restart_required'] is True
    assert result['expected_classes'] == ['ExampleNode'] and 'ready' not in result
    directory = sandbox.base / 'custom_nodes' / sandbox.recipe.folder
    receipt = json.loads((directory / nodes.RECEIPT).read_text(encoding='utf-8'))
    assert receipt['state'] == 'prepared' and receipt['owners'] == ['example.one']
    assert set(receipt['files']) == {'__init__.py', 'install.py'}
    assert pip_installs(sandbox) == []


def test_shared_recipe_is_idempotent_and_keeps_all_owners(sandbox):
    prepare(sandbox)
    assert nodes.plan(sandbox.recipe, owner='example.two')['state'] == 'shared'
    prepare(sandbox, owner='example.two')
    assert nodes.plan(sandbox.recipe, owner='example.one')['state'] == 'current'
    result = prepare(sandbox)
    assert result['state'] == 'prepared' and result['restart_required'] is True
    receipt = json.loads((sandbox.base / 'custom_nodes' / sandbox.recipe.folder / nodes.RECEIPT).read_text())
    assert receipt['owners'] == ['example.one', 'example.two']
    assert len(sandbox.gets) == 1


def test_real_comfy_class_names_and_stable_mixed_case_folder_are_supported(sandbox):
    recipe = replace(sandbox.recipe, folder='ComfyUI_Example', expected_classes=('RIFE VFI', 'Example Node'))
    assert prepare(sandbox, recipe=recipe)['expected_classes'] == ['RIFE VFI', 'Example Node']


@pytest.mark.parametrize('change', [
    {'sha256': 'wrong'}, {'url': 'http://example.org/nodes.zip'},
    {'url': 'https://user:secret@example.org/nodes.zip'}, {'folder': '../escape'},
    {'folder': 'con'}, {'archive_prefix': '../bad'}, {'expected_classes': ()},
    {'schema_version': 2}, {'requirements': ('a>=1',)},
    {'expected_classes': ('Bad\nClass',)}, {'expected_classes': ('  ',)},
    {'requirements': ('a @ https://example.org/a.whl',)}, {'requirements': ('a==1; python_version>"3"',)},
    {'requirements': ('a[extra]==1',)}, {'requirements': ('a==1', 'A==1')},
])
def test_invalid_server_recipes_refused_before_target_operations(sandbox, change):
    with pytest.raises(nodes.NodeInstallError):
        nodes.plan(replace(sandbox.recipe, **change), owner='example.one')
    assert sandbox.calls == [] and sandbox.gets == []


def test_exact_existing_dependencies_do_not_resolve_or_replace_gpu_stack(sandbox):
    recipe = replace(sandbox.recipe, requirements=('torch==2.8.0+cu128', 'numpy==2.2.6'))
    prepare(sandbox, recipe=recipe)
    assert not any('--dry-run' in args for args in sandbox.calls)
    assert pip_installs(sandbox) == []


def test_existing_dependency_conflict_refuses_before_archive_download(sandbox):
    recipe = replace(sandbox.recipe, requirements=('numpy==1.26.4',))
    with pytest.raises(nodes.NodeInstallError, match='conflicts'):
        nodes.plan(recipe, owner='example.one')
    assert sandbox.gets == [] and pip_installs(sandbox) == []


def test_preexisting_broken_environment_is_refused_before_mutation(sandbox):
    sandbox.fail_check = True
    with pytest.raises(nodes.NodeInstallError):
        nodes.plan(replace(sandbox.recipe, requirements=('example-helper==1.2.3',)), owner='example.one')
    assert sandbox.gets == [] and pip_installs(sandbox) == []


def test_missing_wheels_use_constraints_dry_run_hashes_and_comfy_python_only(sandbox):
    result = prepare(sandbox, recipe=replace(sandbox.recipe, requirements=('example-helper==1.2.3',)))
    assert result['packages_to_add'] == [{'name': 'example-helper', 'version': '1.2.3'}]
    assert 'torch==2.8.0+cu128' in sandbox.constraints
    assert 'numpy==2.2.6' in sandbox.constraints
    command = pip_installs(sandbox)[0]
    for option in ('--no-deps', '--no-index', '--only-binary=:all:', '--require-hashes'):
        assert option in command
    assert '--hash=sha256:' + wheel()['download_info']['archive_info']['hashes']['sha256'] in sandbox.locked
    assert sandbox.installed['torch'] == '2.8.0+cu128'


@pytest.mark.parametrize('report', [
    wheel('numpy', '1.26.4'), wheel('torch', '2.9.0+cpu'), wheel('nvidia-cublas-cu12'),
    wheel('xformers'), wheel('bitsandbytes'),
    {**wheel(), 'download_info': {'url': 'https://example.org/pkg.tar.gz', 'archive_info': {'hashes': {'sha256': 'a' * 64}}}},
    {**wheel(), 'download_info': {'url': 'https://example.org/pkg.whl', 'archive_info': {'hashes': {}}}},
])
def test_resolver_cannot_replace_existing_stack_add_gpu_or_build_source(sandbox, report):
    # Include the requested helper so refusal cannot be accidentally explained
    # by an incomplete report instead of the existing-package/GPU guard.
    sandbox.reports = [report] if report['metadata']['name'] == 'example-helper' else [wheel(), report]
    with pytest.raises(nodes.NodeInstallError, match='dependency plan'):
        nodes.plan(replace(sandbox.recipe, requirements=('example-helper==1.2.3',)), owner='example.one')
    assert sandbox.gets == [] and pip_installs(sandbox) == []


def test_hash_failure_has_no_node_or_pip_mutation(sandbox):
    proposal = nodes.plan(sandbox.recipe, owner='example.one')
    sandbox.body = archive_bytes({'__init__.py': 'changed'})
    with pytest.raises(nodes.NodeInstallError, match='hash'):
        nodes.prepare(sandbox.recipe, owner='example.one', plan_id=proposal['plan_id'])
    assert not (sandbox.base / 'custom_nodes').exists() and pip_installs(sandbox) == []


@pytest.mark.parametrize('entries', [
    {'../escape.py': 'bad', '__init__.py': 'ok'},
    {'/absolute.py': 'bad', '__init__.py': 'ok'},
    {'a:stream': 'bad', '__init__.py': 'ok'},
    {'A.py': 'first', 'a.py': 'second', '__init__.py': 'ok'},
    {nodes.RECEIPT: '{}', '__init__.py': 'ok'},
    {'no_entry.py': 'missing init'},
])
def test_archive_paths_metadata_and_entrypoint_are_checked_before_pip(sandbox, entries):
    sandbox.body = archive_bytes(entries)
    recipe = replace(sandbox.recipe, sha256=hashlib.sha256(sandbox.body).hexdigest(),
                     requirements=('example-helper==1.2.3',))
    with pytest.raises(nodes.NodeInstallError, match='archive'):
        prepare(sandbox, recipe=recipe)
    assert not (sandbox.base / 'custom_nodes').exists() and pip_installs(sandbox) == []


def test_archive_symlink_and_zipbomb_are_refused(sandbox):
    for link in (True, False):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('__init__.py', '# node')
            if link:
                info = zipfile.ZipInfo('escape')
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, '../outside')
            else:
                archive.writestr('inflated.txt', '0' * 100000)
        sandbox.body = buffer.getvalue()
        recipe = replace(sandbox.recipe, sha256=hashlib.sha256(sandbox.body).hexdigest())
        with pytest.raises(nodes.NodeInstallError, match='archive'):
            prepare(sandbox, recipe=recipe)
    assert pip_installs(sandbox) == []


def test_explicit_archive_prefix_is_stripped(sandbox):
    sandbox.body = archive_bytes({'release/__init__.py': '# node', 'release/worker.py': '# helper'})
    recipe = replace(sandbox.recipe, sha256=hashlib.sha256(sandbox.body).hexdigest(), archive_prefix='release')
    prepare(sandbox, recipe=recipe)
    assert (sandbox.base / 'custom_nodes' / recipe.folder / 'worker.py').is_file()


def test_foreign_destination_is_never_adopted(sandbox):
    directory = sandbox.base / 'custom_nodes' / sandbox.recipe.folder
    directory.mkdir(parents=True)
    (directory / '__init__.py').write_text('user code')
    with pytest.raises(nodes.NodeInstallError, match='not managed'):
        nodes.plan(sandbox.recipe, owner='example.one')
    assert (directory / '__init__.py').read_text() == 'user code'
    assert sandbox.gets == [] and pip_installs(sandbox) == []


def test_different_shared_version_and_modified_files_refuse(sandbox):
    prepare(sandbox)
    with pytest.raises(nodes.NodeInstallError, match='upgrades are not supported'):
        nodes.plan(replace(sandbox.recipe, version='2.0.0'), owner='example.two')
    path = sandbox.base / 'custom_nodes' / sandbox.recipe.folder / '__init__.py'
    path.write_text('# modified')
    with pytest.raises(nodes.NodeInstallError, match='files changed'):
        nodes.plan(sandbox.recipe, owner='example.one')


@pytest.mark.parametrize('field,value', [('prefix', 'C:/unrelated'), ('executable', 'C:/unrelated/python.exe')])
def test_wrong_python_environment_refused(sandbox, field, value):
    sandbox.probe_override[field] = value
    with pytest.raises(nodes.NodeInstallError, match='own portable interpreter'):
        nodes.plan(sandbox.recipe, owner='example.one')
    assert sandbox.gets == [] and pip_installs(sandbox) == []


def test_nonportable_target_and_remote_endpoint_refused(sandbox):
    sandbox.config['comfyui.api_url'] = 'http://localhost:8188'
    with pytest.raises(nodes.NodeInstallError, match='endpoint'):
        nodes.plan(sandbox.recipe, owner='example.one')
    sandbox.config['comfyui.api_url'] = 'http://127.0.0.1:8188'
    (sandbox.base.parent / 'run_nvidia_gpu.bat').unlink()
    with pytest.raises(nodes.NodeInstallError, match='supported ComfyUI'):
        nodes.plan(sandbox.recipe, owner='example.one')
    assert sandbox.calls == []


def test_plan_becomes_stale_after_environment_change(sandbox):
    proposal = nodes.plan(sandbox.recipe, owner='example.one')
    sandbox.installed['new-user-package'] = '1.0'
    with pytest.raises(nodes.NodeInstallError, match='fresh installation plan'):
        nodes.prepare(sandbox.recipe, owner='example.one', plan_id=proposal['plan_id'])
    assert sandbox.gets == [] and pip_installs(sandbox) == []


def test_plan_cannot_be_reused_for_another_owner(sandbox):
    proposal = nodes.plan(sandbox.recipe, owner='example.one')
    with pytest.raises(nodes.NodeInstallError, match='fresh installation plan'):
        nodes.prepare(sandbox.recipe, owner='example.two', plan_id=proposal['plan_id'])
    assert sandbox.gets == []


def test_pip_failure_is_honest_about_partial_additions_and_never_marks_nodes_prepared(sandbox):
    sandbox.fail_install = True
    with pytest.raises(nodes.NodeInstallError, match='no dependency rollback') as failure:
        prepare(sandbox, recipe=replace(sandbox.recipe, requirements=('example-helper==1.2.3',)))
    assert failure.value.dependencies_may_have_changed is True
    assert 'example-helper' in sandbox.installed
    assert sandbox.installed['torch'] == '2.8.0+cu128'
    assert not (sandbox.base / 'custom_nodes' / sandbox.recipe.folder).exists()


def test_target_changed_during_dependency_install_never_publishes_nodes(sandbox):
    sandbox.after_install = lambda: sandbox.config.update({'comfyui.api_url': 'http://localhost:8188'})
    with pytest.raises(nodes.NodeInstallError) as failure:
        prepare(sandbox, recipe=replace(sandbox.recipe, requirements=('example-helper==1.2.3',)))
    assert failure.value.dependencies_may_have_changed is True
    assert not (sandbox.base / 'custom_nodes' / sandbox.recipe.folder).exists()


def test_other_lds_process_serializes_the_same_comfy_environment(sandbox):
    proposal = nodes.plan(sandbox.recipe, owner='example.one')
    with FileLock(str(sandbox.base.parent / '.lds-node-install.lock'), timeout=0):
        with pytest.raises(nodes.NodeInstallError, match='Another LDS'):
            nodes.prepare(sandbox.recipe, owner='example.one', plan_id=proposal['plan_id'])
    assert sandbox.gets == []


def test_sysconfig_destination_outside_portable_is_refused(sandbox, tmp_path):
    sandbox.probe_override['install_paths'] = {
        'purelib': str(tmp_path / 'another-environment'),
        'platlib': str(sandbox.python.parent / 'Lib' / 'site-packages'),
        'scripts': str(sandbox.python.parent / 'Scripts'), 'data': str(sandbox.python.parent)}
    with pytest.raises(nodes.NodeInstallError):
        nodes.plan(sandbox.recipe, owner='example.one')
    assert sandbox.gets == [] and pip_installs(sandbox) == []


def test_sysconfig_destination_with_linked_ancestor_is_refused(sandbox, tmp_path):
    target = tmp_path / 'other-environment'
    target.mkdir()
    library = sandbox.python.parent / 'Lib'
    library.mkdir()
    site = library / 'site-packages'
    if nodes.os.name == 'nt':
        result = subprocess.run(['cmd', '/c', 'mklink', '/J', str(site), str(target)], capture_output=True)
        assert result.returncode == 0, 'temporary test junction could not be created'
    else:
        site.symlink_to(target, target_is_directory=True)
    try:
        with pytest.raises(nodes.NodeInstallError):
            nodes.plan(sandbox.recipe, owner='example.one')
        assert sandbox.gets == [] and pip_installs(sandbox) == []
    finally:
        if nodes.os.name == 'nt':
            site.rmdir()  # remove the junction itself, never recurse into its target
        else:
            site.unlink()


@pytest.mark.parametrize('changed', ['node', 'receipt'])
def test_existing_node_cannot_adopt_concurrent_edits_during_prepare(sandbox, monkeypatch, changed):
    prepare(sandbox)
    proposal = nodes.plan(sandbox.recipe, owner='example.two')
    folder = sandbox.base / 'custom_nodes' / sandbox.recipe.folder
    marker = folder / nodes.RECEIPT
    before = json.loads(marker.read_text())
    original = nodes._run
    checks = 0

    def race(python, args):
        nonlocal checks
        result = original(python, args)
        if 'check' in args:
            checks += 1
            if checks == 2:
                if changed == 'node':
                    (folder / '__init__.py').write_text('# concurrent change')
                else:
                    marker.write_text(json.dumps({**before, 'owners': ['example.concurrent']}))
        return result

    monkeypatch.setattr(nodes, '_run', race)
    with pytest.raises(nodes.NodeInstallError):
        nodes.prepare(sandbox.recipe, owner='example.two', plan_id=proposal['plan_id'])
    after = json.loads(marker.read_text())
    assert after['files'] == before['files']
    assert 'example.two' not in after['owners']


def test_new_cpu_distribution_cannot_overwrite_gpu_module_files(sandbox):
    sandbox.installed['onnxruntime-gpu'] = '1.22.0'
    library = sandbox.python.parent / 'Lib' / 'site-packages'
    module = library / 'onnxruntime' / '__init__.py'
    module.parent.mkdir(parents=True)
    module.write_text('# existing GPU package')
    body = wheel_bytes('onnxruntime', '1.22.0', files={'onnxruntime/__init__.py': '# CPU replacement'})
    sandbox.reports = [wheel('onnxruntime', '1.22.0', body=body)]
    sandbox.wheel_bodies['onnxruntime'] = body
    with pytest.raises(nodes.NodeInstallError):
        prepare(sandbox, recipe=replace(sandbox.recipe, requirements=('onnxruntime==1.22.0',)))
    assert pip_installs(sandbox) == []
    assert module.read_text() == '# existing GPU package'


def test_new_wheel_script_cannot_replace_existing_command(sandbox):
    scripts = sandbox.python.parent / 'Scripts'
    scripts.mkdir()
    command = scripts / 'shared-command.exe'
    command.write_bytes(b'user command')
    body = wheel_bytes('example-helper', '1.2.3', entry_points='[console_scripts]\nshared-command = example_helper:main\n')
    sandbox.reports = [wheel(body=body)]
    sandbox.wheel_bodies['example-helper'] = body
    with pytest.raises(nodes.NodeInstallError):
        prepare(sandbox, recipe=replace(sandbox.recipe, requirements=('example-helper==1.2.3',)))
    assert pip_installs(sandbox) == [] and command.read_bytes() == b'user command'


@pytest.mark.parametrize('entry', ['shared-command.py', 'pip.py', 'shared command'])
def test_script_name_normalization_cannot_bypass_collision_checks(sandbox, entry):
    body = wheel_bytes('example-helper', '1.2.3', entry_points=f'[console_scripts]\n{entry} = example_helper:main\n')
    sandbox.reports = [wheel(body=body)]
    sandbox.wheel_bodies['example-helper'] = body
    with pytest.raises(nodes.NodeInstallError, match='unsupported paths'):
        prepare(sandbox, recipe=replace(sandbox.recipe, requirements=('example-helper==1.2.3',)))
    assert pip_installs(sandbox) == []


def test_two_new_wheels_cannot_claim_the_same_module(sandbox):
    sandbox.reports = []
    for name in ('first-helper', 'second-helper'):
        body = wheel_bytes(name, '1.0', files={'same_module/__init__.py': name})
        sandbox.reports.append(wheel(name, '1.0', body=body))
        sandbox.wheel_bodies[name] = body
    with pytest.raises(nodes.NodeInstallError, match='same ComfyUI files'):
        prepare(sandbox, recipe=replace(sandbox.recipe, requirements=('first-helper==1.0', 'second-helper==1.0')))
    assert pip_installs(sandbox) == []


@pytest.mark.parametrize('path', ['redirect.pth', 'sitecustomize.py', '../outside.py'])
def test_wheel_interpreter_hooks_and_unsafe_paths_are_refused(sandbox, path):
    body = wheel_bytes('example-helper', '1.2.3', files={path: '# forbidden'})
    sandbox.reports = [wheel(body=body)]
    sandbox.wheel_bodies['example-helper'] = body
    with pytest.raises(nodes.NodeInstallError, match='wheel'):
        prepare(sandbox, recipe=replace(sandbox.recipe, requirements=('example-helper==1.2.3',)))
    assert pip_installs(sandbox) == []


def test_changed_dependency_wheel_is_refused_before_any_pip_install(sandbox):
    sandbox.wheel_bodies['example-helper'] = b'incorrect wheel bytes'
    with pytest.raises(nodes.NodeInstallError, match='hash'):
        prepare(sandbox, recipe=replace(sandbox.recipe, requirements=('example-helper==1.2.3',)))
    assert pip_installs(sandbox) == []


def test_slow_archive_download_is_refused_at_deadline(sandbox, monkeypatch):
    clock = iter((0.0, nodes.MAX_DOWNLOAD_SECONDS + 1.0))
    monkeypatch.setattr(nodes, 'monotonic', lambda: next(clock))
    with pytest.raises(nodes.NodeInstallError, match='time limit'):
        prepare(sandbox)
    assert pip_installs(sandbox) == [] and not (sandbox.base / 'custom_nodes').exists()


def test_continuous_slow_bytes_cannot_hold_a_full_chunk_past_deadline(sandbox, monkeypatch):
    clock = [0.0]
    reads = []

    class SlowResponse:
        status_code = 200
        headers = {}

        def __enter__(self):
            self.raw = self
            return self

        def __exit__(self, *_):
            pass

        def read1(self, size, decode_content=False):
            reads.append(size)
            clock[0] += 20.0
            return b'x'

        def iter_content(self, *_):
            raise AssertionError('a buffering reader would wait for the whole chunk')

    monkeypatch.setattr(nodes, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(nodes.requests, 'get', lambda *a, **kw: SlowResponse())
    with pytest.raises(nodes.NodeInstallError, match='time limit'):
        prepare(sandbox)
    assert 0 < len(reads) <= 7
    assert pip_installs(sandbox) == []


def test_hardlinked_destination_file_refused(sandbox, tmp_path):
    directory = sandbox.base / 'custom_nodes' / sandbox.recipe.folder
    directory.mkdir(parents=True)
    outside = tmp_path / 'outside.json'
    outside.write_text('{}')
    (directory / nodes.RECEIPT).hardlink_to(outside)
    with pytest.raises(nodes.NodeInstallError, match='Hard-linked'):
        nodes.plan(sandbox.recipe, owner='example.one')
    assert outside.read_text() == '{}'


def test_child_command_is_isolated_and_errors_do_not_expose_output(tmp_path, monkeypatch):
    real_run = nodes._run
    captured = []

    def child(command, **kwargs):
        captured.append((command, kwargs))
        return SimpleNamespace(returncode=1, stdout='https://user:password@example.org/private', stderr='secret')

    monkeypatch.setattr(nodes.subprocess, 'run', child)
    monkeypatch.setenv('PIP_TARGET', str(tmp_path / 'wrong'))
    monkeypatch.setenv('PYTHONPATH', str(tmp_path / 'wrong'))
    with pytest.raises(nodes.NodeInstallError) as failure:
        real_run(tmp_path / 'python.exe', ['-m', 'pip', '--isolated', 'check'])
    command, options = captured[0]
    assert command[1:3] == ['-I', '-m']
    assert 'PIP_TARGET' not in options['env'] and 'PYTHONPATH' not in options['env']
    assert options['env']['PYTHONNOUSERSITE'] == '1'
    assert 'password' not in str(failure.value) and 'secret' not in str(failure.value)
