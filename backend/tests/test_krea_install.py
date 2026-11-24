"""Krea 2 Edit — one-click install (weights + custom-node pack).

Krea used to be the ONE engine the app could not install at all: it named four
files and a git repo and left the user to five manual gestures. These tests pin
the two things that silently ruin such an installer — a destination the engine's
own resolver never looks at, and a "downloaded" file that is really a login page
— plus the node-pack path, which is the first third-party CODE this app installs
and therefore the one whose rules are worth enforcing (fixed URL, no shell, no
overwrite, restart required).
"""
import io
import hashlib
import os
import struct
import zipfile

import pytest


@pytest.fixture(autouse=True)
def _reset_runs():
    from app import setup_installer
    setup_installer._runs.clear()
    yield
    setup_installer._runs.clear()


def _make_comfyui(root):
    base = root / 'ComfyUI'
    (base / 'models').mkdir(parents=True, exist_ok=True)
    (base / 'main.py').write_text('# fake ComfyUI entrypoint', encoding='utf-8')
    return base


class _FakeGet:
    """Stand-in for requests.get(..., stream=True) used as a context manager."""
    def __init__(self, status=200, payload=b'', capture=None):
        self._payload = payload
        self.status_code = status
        self.headers = {'content-length': str(len(payload))} if payload else {}
        self._capture = capture

    def __call__(self, url, **kw):
        if self._capture is not None:
            self._capture['url'] = url
            self._capture['headers'] = kw.get('headers')
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_content(self, chunk_size=0):
        if self._payload:
            yield self._payload


def _safetensors(payload_size=1024):
    """Bytes of a minimal but STRUCTURALLY VALID .safetensors file, so a download
    test exercises the real integrity gate instead of tripping it."""
    header = b'{"__metadata__":{"lds":"test"}}'
    return struct.pack('<Q', len(header)) + header + b'\0' * payload_size


KREA_DOWNLOAD_ACTIONS = ('krea_model', 'krea_text_encoder', 'krea_vae',
                         'krea_identity_lora')


def test_install_actions_include_krea_downloads_and_the_node_pack():
    from app import setup_installer
    for a in KREA_DOWNLOAD_ACTIONS + ('krea_nodes',):
        assert a in setup_installer.INSTALL_ACTIONS
        assert a in setup_installer._WORKERS


def test_krea_weights_land_where_the_engine_actually_looks(app, tmp_path):
    """The destination is only right if the ENGINE'S OWN RESOLVERS find the file
    afterwards — a 13 GB download into a folder resolve_krea_unet never scans is
    the exact failure this feature exists to avoid. So: create a file at each
    computed destination, then ask the resolvers."""
    from app import setup_installer, config
    from app.services import krea_edit_helper as keh
    base = _make_comfyui(tmp_path)
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        for action in KREA_DOWNLOAD_ACTIONS:
            dest = setup_installer._download_dest_path(action)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, 'wb') as fh:
                fh.write(b'')
        assert keh.resolve_krea_unet()
        assert keh.resolve_krea_text_encoder() == 'qwen3vl_4b_fp8_scaled.safetensors'
        assert keh.resolve_krea_vae() == 'qwen_image_vae.safetensors'
        assert keh.resolve_krea_identity_lora()[0]
        assert keh.krea_missing_assets() == []


def test_krea_base_model_goes_to_diffusion_models_not_unet(app, tmp_path):
    """`unet` and `diffusion_models` are the same ComfyUI folder TYPE, but
    resolve_krea_unet scans the diffusion_models search roots for a 'krea'-named
    subfolder. Pinned because it is a one-word mistake with a 13 GB cost."""
    from app import setup_installer, config
    base = _make_comfyui(tmp_path)
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        dest = setup_installer._download_dest_path('krea_model')
    assert dest.endswith(os.path.join('models', 'diffusion_models', 'krea',
                                      'krea2_turbo_fp8_scaled.safetensors'))


def test_only_one_krea_base_variant_is_installed():
    """Raw and Turbo are ~13 GB each; the catalog must ship exactly ONE, and it
    must be the build build_workflow's cfg 1.0 / 10-step regime is for."""
    from app import setup_installer
    bases = [k for k, s in setup_installer._KREA_DOWNLOADS.items()
             if s['dest'][0] == 'diffusion_models']
    assert bases == ['krea_model']
    assert 'turbo' in setup_installer._KREA_DOWNLOADS['krea_model']['url'].lower()


def _pin_identity_lora_to(monkeypatch, payload):
    """Point the pinned size and SHA256 of the identity LoRA at a test payload, so
    a download test exercises the real verification instead of tripping it."""
    from app import setup_installer
    spec = setup_installer._MODEL_DOWNLOADS['krea_identity_lora']
    monkeypatch.setitem(spec, 'expected_bytes', len(payload))
    monkeypatch.setitem(spec, 'sha256', hashlib.sha256(payload).hexdigest())


def test_the_identity_lora_is_the_published_file_served_by_the_lds_server():
    """The identity LoRA comes from the LDS download server, pinned to the size
    and SHA256 of the file its author publishes. The author's page stays the
    reference for the terms."""
    from app import setup_installer
    spec = setup_installer._MODEL_DOWNLOADS['krea_identity_lora']
    assert spec['url'] == 'https://lds.perfectgf.fr/models/krea/krea2_identity_edit_v1_2.safetensors'
    assert spec['auth'] == 'none'
    assert spec['expected_bytes'] == 1828256432
    assert spec['sha256'] == '6adf9a69cc9502d286db7b69964d37da7e9cfe4b05b4d004bc275f087d3fd3cf'
    assert spec['license_url'] == 'https://civitai.com/models/2761113'


def test_the_identity_lora_download_sends_no_credential_at_all(app, tmp_path, monkeypatch):
    """Neither the Hugging Face token nor a Civitai key belongs to the LDS
    download server: per-host auth is the rule, and this host takes none."""
    from app import setup_installer, config
    base = _make_comfyui(tmp_path)
    cap, payload = {}, _safetensors()
    monkeypatch.setenv('HF_TOKEN', 'hf_secret_value')
    monkeypatch.setattr(setup_installer, '_civitai_key', lambda: 'civ_key')
    _pin_identity_lora_to(monkeypatch, payload)
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        monkeypatch.setattr(setup_installer.requests, 'get',
                            _FakeGet(payload=payload, capture=cap))
        setup_installer._runs['krea_identity_lora'] = setup_installer._new_run()
        rc = setup_installer._run_model_download('krea_identity_lora')
    assert rc == 0
    assert cap['url'].startswith('https://lds.perfectgf.fr/models/')
    assert not (cap['headers'] or {}).get('Authorization')


def test_a_file_that_is_not_the_published_identity_lora_is_refused(app, tmp_path, monkeypatch):
    """Valid weights that are not the published file (another size, another
    content) must not take the LoRA's place."""
    from app import setup_installer, config
    base = _make_comfyui(tmp_path)
    _pin_identity_lora_to(monkeypatch, _safetensors(2048))
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        monkeypatch.setattr(setup_installer.requests, 'get', _FakeGet(payload=_safetensors()))
        setup_installer._runs['krea_identity_lora'] = setup_installer._new_run()
        rc = setup_installer._run_model_download('krea_identity_lora')
        dest = setup_installer._download_dest_path('krea_identity_lora')
    log = ' '.join(setup_installer._runs['krea_identity_lora']['log'])
    assert rc == 1
    assert not os.path.exists(dest)
    assert 'download verification failed' in log


def test_the_lds_server_refusing_the_identity_lora_asks_for_no_key(app, tmp_path, monkeypatch):
    """A 401/403 from the LDS download server is not a credential problem: the
    message sends nobody to create a Civitai key or a Hugging Face token."""
    from app import setup_installer, config
    base = _make_comfyui(tmp_path)
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        monkeypatch.setattr(setup_installer.requests, 'get', _FakeGet(status=403))
        setup_installer._runs['krea_identity_lora'] = setup_installer._new_run()
        rc = setup_installer._run_model_download('krea_identity_lora')
    log = ' '.join(setup_installer._runs['krea_identity_lora']['log'])
    assert rc == 1
    assert 'LDS download server' in log
    assert 'CIVITAI_API_KEY' not in log and 'HF_TOKEN' not in log


def test_a_civitai_download_keeps_its_own_key_and_never_the_hugging_face_token(monkeypatch):
    """The Civitai provider stays available to a download hosted there: its key
    goes to Civitai only, and the Hugging Face token never does."""
    from app import setup_installer
    monkeypatch.setattr(setup_installer.cfg, 'secret', lambda name: 'hf_secret_value')
    monkeypatch.setattr(setup_installer, '_civitai_key', lambda: 'civ_key')
    assert setup_installer._download_auth({'auth': 'civitai'}) == (
        {'Authorization': 'Bearer civ_key'}, 'civitai')
    monkeypatch.setattr(setup_installer, '_civitai_key', lambda: None)
    assert setup_installer._download_auth({'auth': 'civitai'}) == ({}, 'civitai')


def test_a_login_page_served_as_200_is_rejected_and_deleted(app, tmp_path, monkeypatch):
    """A 200 is not proof of success: auth walls answer with an HTML page carrying
    the right filename, which lands as a perfect-looking .safetensors and kills
    ComfyUI far downstream. It must be refused AND removed — a leftover would make
    every later probe report the asset as installed."""
    from app import setup_installer, config
    base = _make_comfyui(tmp_path)
    html = b'<!doctype html><html><body>Sign in to download</body></html>'
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        monkeypatch.setattr(setup_installer.requests, 'get', _FakeGet(payload=html))
        setup_installer._runs['krea_vae'] = setup_installer._new_run()
        rc = setup_installer._run_model_download('krea_vae')
        dest = setup_installer._download_dest_path('krea_vae')
    log = ' '.join(setup_installer._runs['krea_vae']['log'])
    assert rc == 1
    assert not os.path.exists(dest)
    assert 'not usable weights' in log


def test_a_krea_asset_the_user_already_placed_is_never_re_downloaded(app, tmp_path, monkeypatch):
    """Retrofit: someone who dropped the identity LoRA in loras/ under their own
    name must not watch 1.8 GB download again. The engine's resolver is the
    authority on "installed", not one hardcoded path.

    The fixture is REAL safetensors now, because the resolver's answer alone is no
    longer enough: "it resolves" had to become "it loads" or a hand-placed HTML
    gate page kept certifying itself (test_setup_presence_integrity). A one-byte
    placeholder is, correctly, a corrupted download."""
    import json
    import struct
    from app import setup_installer, config
    base = _make_comfyui(tmp_path)
    mine = base / 'models' / 'loras' / 'my-stuff'
    mine.mkdir(parents=True)
    _hdr = json.dumps({'w': {'dtype': 'F16', 'shape': [1], 'data_offsets': [0, 2]}}).encode()
    (mine / 'krea2_identity_edit_renamed.safetensors').write_bytes(
        struct.pack('<Q', len(_hdr)) + _hdr + b'\x00' * 64)

    def _boom(*a, **kw):
        raise AssertionError('a download was started for an asset already on disk')

    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        monkeypatch.setattr(setup_installer.requests, 'get', _boom)
        setup_installer._runs['krea_identity_lora'] = setup_installer._new_run()
        rc = setup_installer._run_model_download('krea_identity_lora')
    assert rc == 0
    assert any('already installed' in l
               for l in setup_installer._runs['krea_identity_lora']['log'])


def test_install_everything_does_not_silently_pull_20gb_of_krea():
    """Krea installs in one click everywhere it is ASKED for, but it is NOT part
    of the unattended 'Install everything' plan: that would fetch a second ~20 GB
    engine nobody selected."""
    from app import setup_installer
    caps = {'comfyui': {'dir_valid': True,
                        'klein_missing': ['klein_model'],
                        'krea_missing': list(KREA_DOWNLOAD_ACTIONS),
                        'krea_nodes_missing': ['Krea2EditModelPatch']}}
    plan = setup_installer.install_all_plan(caps)
    assert 'klein_model' in plan
    assert not [a for a in plan if a.startswith('krea')]


def test_the_krea_group_is_one_click_and_skips_what_is_there(app):
    """"Everything must be deployable from the app" — so the engine gets ONE
    button that queues the pack and the weights, planned against live gaps so it
    never re-fetches what the user already has."""
    from app import setup_installer
    caps = {'comfyui': {'dir_valid': True, 'reachable': True,
                        'krea_missing': ['krea_model', 'krea_vae'],
                        'krea_nodes_missing': ['Krea2EditModelPatch'],
                        'krea_nodes_installed': False}}
    assert setup_installer.install_group_plan('krea', caps) == [
        'krea_nodes', 'krea_model', 'krea_vae']
    # A pack ComfyUI already exposes (installed under another folder name, e.g.
    # through the ComfyUI Manager) is never cloned a second time.
    assert setup_installer.install_group_plan('krea', {'comfyui': {
        'dir_valid': True, 'reachable': True, 'krea_missing': [],
        'krea_nodes_missing': [], 'krea_nodes_installed': False}}) == []
    # ComfyUI STOPPED: the node probe fails open (nothing reported missing
    # because nothing could be asked). Not on disk + no answer -> still install it,
    # otherwise a one-click install silently drops the pack.
    assert setup_installer.install_group_plan('krea', {'comfyui': {
        'dir_valid': True, 'reachable': False, 'krea_missing': [],
        'krea_nodes_missing': [], 'krea_nodes_installed': False}}) == ['krea_nodes']
    # Pack ON DISK but not loaded = a RESTART, not a re-install: queueing it again
    # would log "already installed" and teach the user nothing.
    caps['comfyui']['krea_nodes_installed'] = True
    assert setup_installer.install_group_plan('krea', caps) == ['krea_model', 'krea_vae']
    # Nothing missing, ComfyUI up -> nothing queued.
    assert setup_installer.install_group_plan(
        'krea', {'comfyui': {'dir_valid': True, 'reachable': True}}) == []
    # No validated ComfyUI -> nowhere to install; never guess a path.
    assert setup_installer.install_group_plan(
        'krea', {'comfyui': {'dir_valid': False, 'krea_missing': ['krea_vae']}}) == []
    assert setup_installer.install_group_plan('nope', {}) == []


def test_install_group_routes(client, monkeypatch):
    from app import setup_installer
    assert client.post('/api/setup/install-group/rm_rf').status_code == 404
    assert client.get('/api/setup/install-group/rm_rf/plan').status_code == 404
    monkeypatch.setattr(setup_installer, 'start_group',
                        lambda g, caps=None: {'plan': ['krea_nodes'], 'statuses': {}})
    r = client.post('/api/setup/install-group/krea')
    assert r.status_code == 200 and r.get_json()['plan'] == ['krea_nodes']


# --- the custom-node pack ---------------------------------------------------

def _zip_bytes(top='comfyui-krea2edit-main'):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        zf.writestr(f'{top}/__init__.py', 'NODE_CLASS_MAPPINGS = {}\n')
    return buf.getvalue()


class _PinnedGet:
    """requests.get serving one pinned body, to the verified worker (raw bytes,
    which never follows a redirect) and to the checked ZIP path (iter_content).
    Integrity rests on the SHA-256, checked on both paths."""
    def __init__(self, payload):
        self.payload = payload
        self.urls = []
        self.redirects = []

    def __call__(self, url, **kw):
        self.redirects.append(kw.get('allow_redirects'))
        self.urls.append(url)
        body = self.payload

        class _Response:
            status_code = 200
            headers = {'Content-Length': str(len(body))}
            raw = None

            def __enter__(self):
                self.raw = self
                self._sent = False
                return self

            def __exit__(self, *a):
                return False

            def read1(self, size, decode_content=False):
                if self._sent:
                    return b''
                self._sent = True
                return body

            def iter_content(self, chunk_size=0):
                yield body
        return _Response()


def _pin_krea_to(monkeypatch, payload, top):
    """The pinned recipe, re-pointed at a synthetic archive with ITS digest."""
    from dataclasses import replace
    from app.services import comfyui_node_recipes
    recipe = replace(comfyui_node_recipes.KREA, sha256=hashlib.sha256(payload).hexdigest(),
                     archive_prefix=top)
    monkeypatch.setitem(comfyui_node_recipes.CORE_RECIPES, 'krea_nodes', recipe)
    return recipe


def test_node_pack_installs_into_this_comfyui_and_demands_a_restart(app, tmp_path, monkeypatch):
    """Installed = placed in THIS install's custom_nodes (found through the same
    resolver every other install uses, never guessed). And success must NOT read
    as "ready": ComfyUI registers nodes at startup only, so the log has to say
    restart — otherwise the engine card stays red and looks like a failed install.
    This layout is not the portable one: a pack that is only files still installs."""
    from app import setup_installer, config
    base = _make_comfyui(tmp_path)
    top = 'comfyui-krea2edit-86f886dac23013d88996e3a2e99093ba44d322fb'
    payload = _zip_bytes(top)
    _pin_krea_to(monkeypatch, payload, top)
    fetch = _PinnedGet(payload)
    monkeypatch.setattr(setup_installer.requests, 'get', fetch)
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        setup_installer._runs['krea_nodes'] = setup_installer._new_run()
        rc = setup_installer._run_node_pack('krea_nodes')
    log = ' '.join(setup_installer._runs['krea_nodes']['log'])
    assert rc == 0, log
    assert (base / 'custom_nodes' / 'comfyui-krea2edit' / '__init__.py').is_file()
    assert 'RESTART ComfyUI' in log
    assert fetch.urls == ['https://lds.perfectgf.fr/models/tools/'
                          'ff1102a1d17f597d8b100afd17bd489a7910325729483398f1385c541f18c171/comfyui-krea2edit.zip']
    # No stray temp folder left next to the pack (a POSIX filelock keeps its
    # inert lock file, which ComfyUI's loader skips).
    from app.services.comfyui_node_install import LOCK_NAME
    assert {p.name for p in (base / 'custom_nodes').iterdir()} - {LOCK_NAME} == {'comfyui-krea2edit'}
    assert sorted(p.name for p in base.iterdir()) == ['custom_nodes', 'main.py', 'models']


def test_node_pack_refuses_an_archive_with_a_changed_checksum(app, tmp_path, monkeypatch):
    from app import setup_installer, config
    base = _make_comfyui(tmp_path)
    monkeypatch.setattr(setup_installer.requests, 'get', _FakeGet(payload=_zip_bytes()))
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        setup_installer._runs['krea_nodes'] = setup_installer._new_run()
        assert setup_installer._run_node_pack('krea_nodes') == 1
    assert 'checksum did not match' in ' '.join(setup_installer._runs['krea_nodes']['log'])
    assert not (base / 'custom_nodes' / 'comfyui-krea2edit').exists()


def test_node_pack_never_clones_a_branch_or_runs_a_command(app, tmp_path, monkeypatch):
    """Third-party code: one pinned archive, checked against its SHA-256. No git
    clone of a moving branch, and on this layout no process at all (no pip)."""
    from app import setup_installer, config
    base = _make_comfyui(tmp_path)
    top = 'comfyui-krea2edit-86f886dac23013d88996e3a2e99093ba44d322fb'
    payload = _zip_bytes(top)
    _pin_krea_to(monkeypatch, payload, top)
    monkeypatch.setattr(setup_installer.requests, 'get', _PinnedGet(payload))

    def _no_process(*a, **kw):
        raise AssertionError(f'a node pack install ran a process: {a}')

    monkeypatch.setattr(setup_installer.subprocess, 'run', _no_process)
    monkeypatch.setattr(setup_installer.subprocess, 'Popen', _no_process)
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        setup_installer._runs['krea_nodes'] = setup_installer._new_run()
        rc = setup_installer._run_node_pack('krea_nodes')
    assert rc == 0
    assert not hasattr(setup_installer, '_clone_node_pack')


def test_node_pack_never_overwrites_an_existing_install(app, tmp_path, monkeypatch):
    """Someone may have patched the pack or pinned a commit through the ComfyUI
    Manager. Re-clicking Install must be a no-op, not a wipe."""
    from app import setup_installer, config
    base = _make_comfyui(tmp_path)
    dest = base / 'custom_nodes' / 'comfyui-krea2edit'
    dest.mkdir(parents=True)
    (dest / '__init__.py').write_text('# my patched copy', encoding='utf-8')

    def _boom(*a, **kw):
        raise AssertionError('an existing node pack was overwritten')

    monkeypatch.setattr(setup_installer.subprocess, 'run', _boom)
    monkeypatch.setattr(setup_installer.requests, 'get', _boom)
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        setup_installer._runs['krea_nodes'] = setup_installer._new_run()
        rc = setup_installer._run_node_pack('krea_nodes')
    assert rc == 0
    assert (dest / '__init__.py').read_text(encoding='utf-8') == '# my patched copy'


def test_node_pack_without_a_known_comfyui_fails_by_saying_so(app, tmp_path, monkeypatch):
    """Never clone at a guessed location: ComfyUI lives somewhere different on
    every machine (portable, Desktop, Linux). No valid folder -> a refusal that
    explains itself, and no traceback."""
    from app import setup_installer, config

    def _boom(*a, **kw):
        raise AssertionError('cloned without a validated ComfyUI folder')

    monkeypatch.setattr(setup_installer.subprocess, 'run', _boom)
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(tmp_path / 'nope')}})
        setup_installer._runs['krea_nodes'] = setup_installer._new_run()
        rc = setup_installer._run_node_pack('krea_nodes')
    log = ' '.join(setup_installer._runs['krea_nodes']['log'])
    assert rc == 1
    assert 'ComfyUI' in log and 'nothing was installed' in log


def test_node_pack_failure_tells_the_user_what_to_do_by_hand(app, tmp_path, monkeypatch):
    """git missing AND the ZIP unreachable is a real state (offline, proxy,
    corporate TLS). It must end on instructions, not a bare error, and leave
    nothing half-installed for ComfyUI to try to import."""
    from app import setup_installer, config
    base = _make_comfyui(tmp_path)
    monkeypatch.setattr(setup_installer.shutil, 'which', lambda n: None)
    monkeypatch.setattr(setup_installer.requests, 'get', _FakeGet(status=404))
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        setup_installer._runs['krea_nodes'] = setup_installer._new_run()
        rc = setup_installer._run_node_pack('krea_nodes')
    log = ' '.join(setup_installer._runs['krea_nodes']['log'])
    assert rc == 1
    assert 'lds.perfectgf.fr/models/tools/' in log and 'comfyui-krea2edit.zip' in log
    assert 'ComfyUI Manager' in log
    assert not (base / 'custom_nodes' / 'comfyui-krea2edit').exists()


def test_the_pack_on_disk_is_distinguished_from_the_pack_being_loaded(app, tmp_path):
    """The last inch: after the install, /object_info still reports the nodes as
    missing until ComfyUI restarts. Without a separate "on disk" signal the app
    would tell the user to install what they just watched install."""
    from app import config
    from app.services import krea_edit_helper as keh
    base = _make_comfyui(tmp_path)
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        assert keh.krea_node_pack_installed() is False
        pack = base / 'custom_nodes' / 'comfyui-krea2edit'
        pack.mkdir(parents=True)
        (pack / '__init__.py').write_text('# nodes', encoding='utf-8')
        assert keh.krea_node_pack_installed() is True


def test_krea_node_cache_is_cleared_after_an_install(app):
    """The node probe caches SUCCESS for 5 minutes. Installing the pack must not
    leave a stale verdict deciding what the engine card says."""
    import time
    from app.services import krea_edit_helper as keh
    keh._nodes_ok_until = time.time() + 999
    keh.clear_nodes_cache()
    assert keh._nodes_ok_until == 0.0
