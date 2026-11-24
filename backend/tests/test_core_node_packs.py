"""The core's ComfyUI node packs: one pinned, verified archive each, on any layout.

A pack that is only files installs into the configured ComfyUI whatever its
layout. A pack that adds Python packages installs by itself only on the Windows
portable whose Python LDS manages; elsewhere the action installs nothing and
returns the exact steps. No network, no git, no pip and no real ComfyUI here.
"""
from dataclasses import replace
import hashlib
import io
import zipfile

import pytest

from app import setup_installer as installer
from app.services import comfyui_node_install as nodes
from app.services import comfyui_node_recipes as recipes

# The pins, each archive downloaded twice and hashed (2026-11-24).
PINS = {
    'krea_nodes': ('lbouaraba/comfyui-krea2edit', '86f886dac23013d88996e3a2e99093ba44d322fb',
                   'ff1102a1d17f597d8b100afd17bd489a7910325729483398f1385c541f18c171'),
    'lanpaint_nodes': ('scraed/LanPaint', '2d7912f9a5efe5ece8de334c7ca18317b8288c39',
                       'bc79a3466fb607d7ee70076b03059d11c14391cee25ad40f6afefc4604628d2f'),
    'detail_daemon_nodes': ('Jonseed/ComfyUI-Detail-Daemon', '3394e44afea04ed0188fb37b21f0d9952469766b',
                            '5f0eaa7774d5d9d3daccdb5e77b97f416dfadcfcf52ecb14205ce3c0784d6c6d'),
}


@pytest.fixture(autouse=True)
def _reset_runs():
    installer._runs.clear()
    yield
    installer._runs.clear()


@pytest.fixture
def comfy(app, tmp_path, monkeypatch):
    """A manual/Desktop-style ComfyUI (not the portable), configured in LDS."""
    from app import config
    base = tmp_path / 'ComfyUI'
    (base / 'models').mkdir(parents=True)
    (base / 'main.py').write_text('# not a real ComfyUI', encoding='utf-8')

    def _no_process(*args, **kwargs):
        raise AssertionError(f'a node pack install ran a process: {args}')

    monkeypatch.setattr(installer.subprocess, 'run', _no_process)
    monkeypatch.setattr(installer.subprocess, 'Popen', _no_process)
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        yield base


def _archive(top):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr(f'{top}/__init__.py', 'NODE_CLASS_MAPPINGS = {}\n')
        archive.writestr(f'{top}/nodes.py', '# synthetic\n')
    return buffer.getvalue()


class _Serve:
    """requests.get serving one body, for the verified worker and the ZIP path."""
    def __init__(self, body):
        self.body = body
        self.urls = []

    def __call__(self, url, **kwargs):
        self.urls.append(url)
        body = self.body

        class _Response:
            status_code = 200
            headers = {'Content-Length': str(len(body))}

            def __enter__(self):
                self.raw, self._sent = self, False
                return self

            def __exit__(self, *_):
                return False

            def read1(self, size, decode_content=False):
                sent, self._sent = self._sent, True
                return b'' if sent else body

            def iter_content(self, chunk_size=0):
                yield body
        return _Response()


def _run(action):
    installer._runs[action] = installer._new_run()
    return installer._run_node_pack(action), installer._runs[action]


def _pack_names(base):
    folder = base / 'custom_nodes'
    return {p.name for p in folder.iterdir()} - {nodes.LOCK_NAME} if folder.exists() else set()


def test_every_core_pack_is_pinned_to_one_verified_commit_archive():
    assert set(installer._NODE_PACKS) == set(PINS)
    for action, (repo, commit, sha256) in PINS.items():
        spec = installer._NODE_PACKS[action]
        recipe = recipes.CORE_RECIPES.get(action)
        assert spec['repo'] == f'https://github.com/{repo}'
        assert 'refs/heads' not in repr(spec) and 'clone' not in repr(spec)
        if recipe is None:
            assert (spec['commit'], spec['zip'], spec['sha256']) == (
                commit, f"https://lds.perfectgf.fr/models/tools/{sha256}/{repo.split('/')[1]}.zip", sha256)
            continue
        assert (recipe.url, recipe.sha256) == (f"https://lds.perfectgf.fr/models/tools/{sha256}/{repo.split('/')[1]}.zip", sha256)
        assert recipe.archive_prefix == f"{repo.split('/')[1]}-{commit}"
        assert recipe.folder == spec['folder']
        nodes._validate(recipe, 'lds.core')
    assert recipes.KREA.requirements == () and recipes.KREA.version == '1.2.5'
    assert recipes.DETAIL_DAEMON.requirements == ('matplotlib==3.11.2',)
    assert 'detail_daemon_nodes' in installer.INSTALL_ACTIONS
    assert not hasattr(installer, '_clone_node_pack')


@pytest.mark.parametrize('action', ['krea_nodes', 'lanpaint_nodes'])
def test_an_archive_with_another_sha256_is_refused_and_custom_nodes_stays_empty(comfy, monkeypatch, action):
    top = f"{PINS[action][0].split('/')[1]}-{PINS[action][1]}"
    serve = _Serve(_archive(top))      # the right layout, but not the pinned bytes
    monkeypatch.setattr(installer.requests, 'get', serve)
    rc, run = _run(action)
    log = '\n'.join(run['log'])
    assert rc == 1
    assert 'hash does not match' in log or 'checksum did not match' in log
    assert _pack_names(comfy) == set()
    assert f"Download https://lds.perfectgf.fr/models/tools/{PINS[action][2]}/{PINS[action][0].split('/')[1]}.zip" in log
    assert sorted(p.name for p in comfy.iterdir()) == ['custom_nodes', 'main.py', 'models']


def test_lanpaint_installs_its_pinned_archive_on_a_non_portable_layout(comfy, monkeypatch):
    top = f'LanPaint-{PINS["lanpaint_nodes"][1]}'
    body = _archive(top)
    monkeypatch.setitem(installer._NODE_PACKS['lanpaint_nodes'], 'sha256', hashlib.sha256(body).hexdigest())
    serve = _Serve(body)
    monkeypatch.setattr(installer.requests, 'get', serve)
    rc, run = _run('lanpaint_nodes')
    assert rc == 0, run['log']
    assert serve.urls == [installer._NODE_PACKS['lanpaint_nodes']['zip']]
    assert (comfy / 'custom_nodes' / 'LanPaint' / 'nodes.py').is_file()
    assert _pack_names(comfy) == {'LanPaint'}
    assert 'RESTART ComfyUI' in '\n'.join(run['log'])


def test_krea_installs_its_pinned_archive_on_a_non_portable_layout(comfy, monkeypatch):
    """A files-only pack on a layout whose Python LDS does not manage takes the
    recipe's pinned archive straight into custom_nodes, like the former clone did,
    without the managed worker (whose link checks refused custom_nodes folders
    reached through a link, which the clone accepted)."""
    body = _archive(recipes.KREA.archive_prefix)
    monkeypatch.setitem(recipes.CORE_RECIPES, 'krea_nodes',
                        replace(recipes.KREA, sha256=hashlib.sha256(body).hexdigest()))
    serve = _Serve(body)
    monkeypatch.setattr(installer.requests, 'get', serve)
    monkeypatch.setattr(nodes, 'plan', lambda *a, **k: pytest.fail('the managed worker ran off the portable'))
    rc, run = _run('krea_nodes')
    assert rc == 0, run['log']
    assert serve.urls == [recipes.KREA.url]
    pack = comfy / 'custom_nodes' / 'comfyui-krea2edit'
    assert (pack / '__init__.py').is_file() and not (pack / nodes.RECEIPT).exists()
    assert _pack_names(comfy) == {'comfyui-krea2edit'}
    assert 'RESTART ComfyUI' in '\n'.join(run['log'])


def test_a_prepared_krea_copy_off_the_portable_is_left_alone_not_unpacked_into(comfy, monkeypatch):
    """On a layout the worker does not own, a folder already there is never
    re-prepared nor written into, even when its receipt names the current version."""
    import json
    folder = comfy / 'custom_nodes' / 'comfyui-krea2edit'
    folder.mkdir(parents=True)
    (folder / '__init__.py').write_text('# prepared earlier', encoding='utf-8')
    (folder / nodes.RECEIPT).write_text(json.dumps(
        {'recipe_id': 'comfyui-krea2edit', 'version': recipes.KREA.version, 'state': 'prepared'}),
        encoding='utf-8')
    monkeypatch.setattr(nodes, 'plan', lambda *a, **k: pytest.fail('re-prepared off the portable'))
    monkeypatch.setattr(installer.requests, 'get', lambda *a, **k: pytest.fail('downloaded over a folder'))
    rc, run = _run('krea_nodes')
    assert rc == 0 and 'left untouched' in '\n'.join(run['log'])
    assert sorted(p.name for p in folder.iterdir()) == sorted(['__init__.py', nodes.RECEIPT])


def test_detail_daemon_on_a_non_portable_layout_returns_the_exact_steps_and_runs_no_pip(comfy, monkeypatch):
    def _no_download(*args, **kwargs):
        raise AssertionError('nothing may be downloaded for a pack LDS will not install')

    monkeypatch.setattr(installer.requests, 'get', _no_download)
    rc, run = _run('detail_daemon_nodes')
    repo, commit, sha256 = PINS['detail_daemon_nodes']
    expected = [
        f"Download https://lds.perfectgf.fr/models/tools/{sha256}/{repo.split('/')[1]}.zip",
        f'Check that the downloaded file has the SHA-256 {sha256}',
        f'Extract the folder ComfyUI-Detail-Daemon-{commit} from it into {comfy / "custom_nodes"} '
        'and rename it to ComfyUI-Detail-Daemon',
        'If ComfyUI’s Python does not have matplotlib yet, add it, with the Python that starts '
        'ComfyUI: python -m pip install "matplotlib==3.11.2"',
        'Restart ComfyUI.',
    ]
    assert rc == 1
    assert run['manual_steps'] == expected
    assert installer.status('detail_daemon_nodes')['manual_steps'] == expected
    assert 'installs those automatically only on a ComfyUI Windows portable' in run['log'][0]
    assert _pack_names(comfy) == set()


def test_detail_daemon_steps_name_the_python_a_desktop_install_runs(comfy, monkeypatch):
    python = comfy / '.venv' / 'Scripts' / 'python.exe'
    python.parent.mkdir(parents=True)
    python.write_bytes(b'never executed')
    rc, run = _run('detail_daemon_nodes')
    assert rc == 1
    assert run['manual_steps'][3] == ('If ComfyUI’s Python does not have matplotlib yet, add it: '
                                      f'"{python}" -m pip install "matplotlib==3.11.2"')


def test_the_portable_path_hands_the_pinned_requirement_to_the_managed_worker(comfy, monkeypatch):
    seen = []

    def plan(recipe, *, owner):
        seen.append(('plan', recipe, owner))
        return {'plan_id': 'reviewed', 'state': 'install'}

    def prepare(recipe, *, owner, plan_id, log):
        seen.append(('prepare', recipe, owner, plan_id))
        return {'state': 'prepared', 'restart_required': True}

    monkeypatch.setattr(nodes, 'plan', plan)
    monkeypatch.setattr(nodes, 'prepare', prepare)
    rc, run = _run('detail_daemon_nodes')
    assert rc == 0
    assert [step[0] for step in seen] == ['plan', 'prepare']
    recipe = seen[0][1]
    assert recipe is recipes.DETAIL_DAEMON and recipe.requirements == ('matplotlib==3.11.2',)
    assert seen[1][2:] == ('lds.core', 'reviewed')
    assert 'RESTART ComfyUI when it is idle' in '\n'.join(run['log'])


@pytest.mark.parametrize('action', ['krea_nodes', 'lanpaint_nodes', 'detail_daemon_nodes'])
def test_a_pack_folder_already_there_is_left_alone(comfy, monkeypatch, action):
    folder = comfy / 'custom_nodes' / installer._NODE_PACKS[action]['folder']
    folder.mkdir(parents=True)
    (folder / '__init__.py').write_text('# my own copy', encoding='utf-8')
    monkeypatch.setattr(nodes, 'plan', lambda *a, **k: pytest.fail('an existing folder was planned'))
    monkeypatch.setattr(installer.requests, 'get', lambda *a, **k: pytest.fail('downloaded over a folder'))
    rc, run = _run(action)
    assert rc == 0 and 'left untouched' in '\n'.join(run['log'])
    assert (folder / '__init__.py').read_text(encoding='utf-8') == '# my own copy'


def test_an_older_prepared_krea_copy_is_left_alone_instead_of_an_upgrade_error(comfy, monkeypatch):
    import json
    folder = comfy / 'custom_nodes' / 'comfyui-krea2edit'
    folder.mkdir(parents=True)
    (folder / '__init__.py').write_text('# 1.2.0', encoding='utf-8')
    (folder / nodes.RECEIPT).write_text(json.dumps(
        {'recipe_id': 'comfyui-krea2edit', 'version': '1.2.0', 'state': 'prepared'}), encoding='utf-8')
    monkeypatch.setattr(nodes, 'plan', lambda *a, **k: pytest.fail('a supported copy was asked to upgrade'))
    rc, run = _run('krea_nodes')
    assert rc == 0 and 'left untouched' in '\n'.join(run['log'])


def test_the_sdxl_studio_offers_the_detail_daemon_install_where_the_node_is_missing(app):
    from app.routes._common import _studio_missing_response
    from app.services.lora_test_studio import StudioAssetsMissing, studio_missing_node_hints
    [hint] = studio_missing_node_hints(['DetailDaemonSamplerNode'])
    assert hint['install_action'] == 'detail_daemon_nodes'
    assert installer.known_action(hint['install_action'])
    with app.test_request_context():
        response, status = _studio_missing_response(
            StudioAssetsMissing('sdxl', [], ['DetailDaemonSamplerNode']))
    body = response.get_json()
    assert status == 409
    assert body['studio_missing']['node_packs'][0]['install_action'] == 'detail_daemon_nodes'
    assert 'install “ComfyUI-Detail-Daemon” in one click under “Fix missing ComfyUI nodes”' in body['error']


@pytest.mark.parametrize('catalog,expected', [
    ({'h3_turbo_nodes': {'kind': 'node_pack', 'plugin': 'video'}}, 'h3_turbo_nodes'),
    ({'live_h3_turbo_nodes': {'kind': 'node_pack', 'plugin': 'live'}}, 'live_h3_turbo_nodes'),
    ({}, None),                                           # a plugin that installs nothing
    ({'h3_turbo_nodes': {'kind': 'download', 'plugin': 'video'}}, None),
    (RuntimeError('registry unreadable'), None),
])
def test_a_video_pack_points_to_setup_only_when_a_plugin_installs_it(app, monkeypatch, catalog, expected):
    from app.routes._common import _studio_missing_response
    from app.services.lora_test_studio import StudioAssetsMissing, studio_missing_node_hints

    def plugin_actions_catalog():
        if isinstance(catalog, Exception):
            raise catalog
        return catalog

    monkeypatch.setattr(installer, 'plugin_actions_catalog', plugin_actions_catalog)
    [hint] = studio_missing_node_hints(['MiniMaxH3TurboSampler'])
    assert hint.get('setup_action') == expected and 'plugin_actions' not in hint
    with app.test_request_context():
        response, _status = _studio_missing_response(
            StudioAssetsMissing('video', [], ['MiniMaxH3TurboSampler']))
    error = response.get_json()['error']
    if expected:
        assert 'install “ComfyUI-MiniMax-H3-Turbo” in one click from Setup' in error
        assert 'ComfyUI-Manager' not in error
    else:
        assert 'install “ComfyUI-MiniMax-H3-Turbo” via ComfyUI-Manager' in error
