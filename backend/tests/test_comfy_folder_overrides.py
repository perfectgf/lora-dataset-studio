"""ComfyUI folder overrides — the four `comfyui.*_dir` keys.

Reported on Discord by vykas22: a ComfyUI started with custom input/output folders
looked like it was ignored. The keys always worked; nothing in the app let you set
them, and nothing showed which folder was actually in use.

These tests hold the line on all three halves of that: the override really reaches
the code that reads/writes those folders (not merely stored), an empty override keeps
the historical derived behaviour byte for byte, and a path that isn't on disk is
reported rather than swallowed.
"""
import importlib

import pytest


def _fresh(monkeypatch, tmp_path):
    monkeypatch.setenv('LDS_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('LDS_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setenv('LDS_ENV', str(tmp_path / '.env'))
    import app.config as config
    importlib.reload(config)
    return config


# --- resolution ------------------------------------------------------------------

def test_override_wins_over_derived(tmp_path, monkeypatch):
    config = _fresh(monkeypatch, tmp_path)
    base, custom_in, custom_out = tmp_path / 'Comfy', tmp_path / 'in', tmp_path / 'out'
    config.save_config({'comfyui': {'base_dir': str(base),
                                    'input_dir': str(custom_in),
                                    'output_dir': str(custom_out)}})
    assert config.comfyui_dir('input') == custom_in
    assert config.comfyui_dir('output') == custom_out
    # untouched kinds still derive
    assert config.comfyui_dir('models') == base / 'models'
    assert config.comfyui_dir('loras') == base / 'models' / 'loras'


def test_empty_override_keeps_historical_derivation(tmp_path, monkeypatch):
    """Non-regression: blank fields must behave exactly as before they existed."""
    config = _fresh(monkeypatch, tmp_path)
    base = tmp_path / 'Comfy'
    config.save_config({'comfyui': {'base_dir': str(base), 'input_dir': '',
                                    'output_dir': '', 'models_dir': '', 'loras_dir': ''}})
    assert config.comfyui_dir('input') == base / 'input'
    assert config.comfyui_dir('output') == base / 'output'
    assert config.comfyui_dir('models') == base / 'models'
    assert config.comfyui_dir('loras') == base / 'models' / 'loras'


def test_whitespace_override_is_not_a_path(tmp_path, monkeypatch):
    """A stray space used to resolve to Path(' ') and shadow the derived folder."""
    config = _fresh(monkeypatch, tmp_path)
    base = tmp_path / 'Comfy'
    config.save_config({'comfyui': {'base_dir': str(base), 'output_dir': '   '}})
    assert config.comfyui_dir('output') == base / 'output'


def test_override_without_base_dir_still_resolves(tmp_path, monkeypatch):
    """An override is self-sufficient: pointing only at an output folder works even
    with no install directory (nothing to derive from)."""
    config = _fresh(monkeypatch, tmp_path)
    config.save_config({'comfyui': {'base_dir': '', 'output_dir': str(tmp_path / 'out')}})
    assert config.comfyui_dir('output') == tmp_path / 'out'
    assert config.comfyui_dir('input') is None


def test_legacy_handwritten_config_json_still_honoured(tmp_path, monkeypatch):
    """Installs that set these keys by hand (the only way until now) must keep
    working untouched — the UI writes the same keys, it does not replace them."""
    import json
    cfg_path = tmp_path / 'config.json'
    cfg_path.write_text(json.dumps({'comfyui': {'base_dir': str(tmp_path / 'Comfy'),
                                                'input_dir': str(tmp_path / 'legacy-in'),
                                                'output_dir': str(tmp_path / 'legacy-out')}}),
                        encoding='utf-8')
    config = _fresh(monkeypatch, tmp_path)
    assert config.comfyui_dir('input') == tmp_path / 'legacy-in'
    assert config.comfyui_dir('output') == tmp_path / 'legacy-out'


def test_resolve_is_pure_and_independent_of_saved_config(tmp_path, monkeypatch):
    """The Settings preview resolves UNSAVED values through this same function —
    that identity is what stops the preview drifting from real behaviour."""
    from pathlib import Path
    config = _fresh(monkeypatch, tmp_path)
    here, other = str(tmp_path / 'base'), str(tmp_path / 'elsewhere')
    assert config.resolve_comfyui_dir('input', here, '') == Path(here) / 'input'
    assert config.resolve_comfyui_dir('input', here, other) == Path(other)
    assert config.resolve_comfyui_dir('output', '', '') is None


# --- the override actually reaches the consumers ----------------------------------

CONSUMERS_INPUT = [
    ('app.services.klein_edit_helper', '_comfy_input_dir'),
    ('app.services.watermark_klein', '_comfy_input_dir'),
]
CONSUMERS_OUTPUT = [
    ('app.services.klein_edit_helper', '_comfy_output_dir'),
    ('app.services.watermark_klein', '_comfy_output_dir'),
    ('app.services.lora_test_studio', '_comfy_output_dir'),
    ('app.services.face_dataset_service', '_comfy_output_dir'),
    ('app.utils.comfyui', '_out_dir'),
]


@pytest.mark.parametrize('module_name, func_name', CONSUMERS_INPUT + CONSUMERS_OUTPUT)
def test_consumer_uses_the_override(module_name, func_name, tmp_path, monkeypatch, app):
    """Every place that reads a ComfyUI input/output folder must land on the override,
    not on <base>/input|output. These read config LIVE (documented in each module), so
    patching the resolver is enough — no restart, no import order to respect."""
    module = importlib.import_module(module_name)
    kind = 'input' if func_name.endswith('input_dir') else 'output'
    custom = tmp_path / f'custom-{kind}'
    with app.app_context():
        monkeypatch.setattr('app.config.comfyui_dir',
                            lambda k: custom if k == kind else tmp_path / 'derived')
        assert getattr(module, func_name)() == str(custom)


@pytest.mark.parametrize('module_name, func_name', CONSUMERS_INPUT + CONSUMERS_OUTPUT)
def test_consumer_reads_config_live_end_to_end(module_name, func_name, tmp_path, monkeypatch, app):
    """The same check driven from config.json rather than a patched resolver, so the
    whole chain (saved key -> comfyui_dir -> consumer) is exercised at least once."""
    import app.config as config
    module = importlib.import_module(module_name)
    kind = 'input' if func_name.endswith('input_dir') else 'output'
    custom = tmp_path / f'e2e-{kind}'
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(tmp_path / 'Comfy'),
                                        f'{kind}_dir': str(custom)}})
        assert getattr(module, func_name)() == str(custom)


def test_model_paths_default_roots_follow_the_overrides(tmp_path, monkeypatch, app):
    """models_dir/loras_dir must move the DEFAULT model roots, while
    extra_model_paths.yaml keeps being located from base_dir (same tree as ComfyUI
    reads it from) — the two mechanisms stack, they don't fight."""
    import app.config as config
    from app.services import comfy_model_paths as cmp
    base, models, loras = tmp_path / 'Comfy', tmp_path / 'M', tmp_path / 'L'
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base), 'models_dir': str(models),
                                        'loras_dir': str(loras)}})
        assert cmp._default_roots('loras') == [str(loras)]
        assert cmp._default_roots('checkpoints') == [str(models / 'checkpoints')]
        # yaml still sought next to main.py, NOT under the models override
        assert cmp._yaml_path() == str(base / cmp.YAML_FILENAME)


# --- classification: what the Settings fields show --------------------------------

def test_classify_reports_derived_paths_and_existence(tmp_path):
    from app import capabilities
    base = tmp_path / 'Comfy'
    (base / 'output').mkdir(parents=True)
    r = capabilities.classify_comfyui_folders(str(base), {})
    assert r['output_dir'] == {'kind': 'output', 'source': 'derived',
                               'resolved': str(base / 'output'), 'exists': True,
                               # readable from here — the OTHER half of the contract
                               # (see test_comfy_input_folder_handoff.py)
                               'usable': True, 'problem': ''}
    # input/ was never created -> the field must say so rather than look fine
    assert r['input_dir']['source'] == 'derived'
    assert r['input_dir']['exists'] is False


def test_classify_flags_a_typed_path_that_does_not_exist(tmp_path):
    from app import capabilities
    ghost = tmp_path / 'nope'
    r = capabilities.classify_comfyui_folders(str(tmp_path / 'Comfy'),
                                              {'input_dir': str(ghost)})
    assert r['input_dir']['source'] == 'override'
    assert r['input_dir']['resolved'] == str(ghost)
    assert r['input_dir']['exists'] is False


def test_classify_unset_when_nothing_to_resolve():
    from app import capabilities
    r = capabilities.classify_comfyui_folders('', {})
    for info in r.values():
        assert info == {'kind': info['kind'], 'source': 'unset', 'resolved': '',
                        'exists': False, 'usable': None, 'problem': ''}


def test_classify_covers_all_four_kinds(tmp_path):
    from app import capabilities
    from app import config
    r = capabilities.classify_comfyui_folders(str(tmp_path), {})
    assert set(r) == {'output_dir', 'input_dir', 'models_dir', 'loras_dir'}
    assert len(config.COMFY_DIR_KINDS) == 4


# --- detection from ComfyUI's own command line ------------------------------------

def test_parse_argv_reads_both_argparse_spellings():
    from app.capabilities import parse_comfy_argv_dirs
    argv = ['main.py', '--listen', '--input-directory', r'D:\comfy-in',
            r'--output-directory=D:\comfy-out', '--models-directory', r'E:\models']
    assert parse_comfy_argv_dirs(argv) == {'input_dir': r'D:\comfy-in',
                                           'output_dir': r'D:\comfy-out',
                                           'models_dir': r'E:\models'}


def test_parse_argv_drops_relative_paths():
    """Relative paths resolve against ComfyUI's working directory, which we do not
    know. Offering one would be a guess — the rule is ask, never assume."""
    from app.capabilities import parse_comfy_argv_dirs
    assert parse_comfy_argv_dirs(['main.py', '--output-directory', 'my_outputs']) == {}
    assert parse_comfy_argv_dirs(['main.py', '--output-directory=../out']) == {}


def test_parse_argv_reads_a_container_path_whatever_the_host_is():
    r"""The case these fields exist for: ComfyUI in WSL or a container, reporting
    `/workspace/…` to a Windows reader.

    `os.path.isabs` answers False for that on Windows — a leading slash is
    drive-relative there, and Python 3.13 made the rule explicit — so the
    suggestion vanished on exactly the installs that need it. Where a POSIX path
    did get through, `os.path.normpath` then rewrote `/mnt/shared/input` as
    `\mnt\shared\input`, a path correct on neither side. Both are judged under the
    convention the PATH uses now, so this passes on either host."""
    from app.capabilities import parse_comfy_argv_dirs
    assert parse_comfy_argv_dirs(['main.py', '--input-directory', '/workspace/in',
                                  '--output-directory=/mnt/shared/out']) == {
        'input_dir': '/workspace/in', 'output_dir': '/mnt/shared/out'}
    # ...and the Windows spelling survives a POSIX reader, for the same reason.
    assert parse_comfy_argv_dirs(['main.py', '--models-directory', 'E:\\models']) == {
        'models_dir': 'E:\\models'}


def test_parse_argv_ignores_base_directory():
    """--base-directory is a layout, not an answer: the install-directory field
    already derives from it, and turning it into input/output would be an assumption."""
    from app.capabilities import parse_comfy_argv_dirs
    assert parse_comfy_argv_dirs(['main.py', '--base-directory', r'D:\comfy']) == {}


def test_parse_argv_tolerates_junk():
    from app.capabilities import parse_comfy_argv_dirs
    assert parse_comfy_argv_dirs(None) == {}
    assert parse_comfy_argv_dirs([]) == {}
    assert parse_comfy_argv_dirs(['main.py']) == {}
    assert parse_comfy_argv_dirs(['--input-directory']) == {}            # flag, no value
    assert parse_comfy_argv_dirs(['--input-directory', '--listen']) == {}  # value is a flag


def test_detect_returns_empty_when_comfyui_says_nothing(monkeypatch, app):
    """No argv field (older ComfyUI), unreachable, or launched plainly -> {}.
    An empty dict means "nothing to offer", never "use these defaults"."""
    from app import capabilities
    import app.config as config

    class _Resp:
        status_code = 200

        def __init__(self, payload):
            self._p = payload

        def json(self):
            return self._p

    with app.app_context():
        config.save_config({'comfyui': {'api_url': 'http://127.0.0.1:9'}})
        monkeypatch.setattr(capabilities.requests, 'get', lambda *a, **k: _Resp({'system': {}}))
        assert capabilities.detect_comfyui_folders() == {}

        def _boom(*a, **k):
            raise OSError('unreachable')
        monkeypatch.setattr(capabilities.requests, 'get', _boom)
        assert capabilities.detect_comfyui_folders() == {}


def test_detect_reads_argv_from_system_stats(monkeypatch, app):
    from app import capabilities
    import app.config as config

    class _Resp:
        status_code = 200

        def json(self):
            return {'system': {'argv': ['main.py', '--output-directory', r'D:\out']}}

    with app.app_context():
        config.save_config({'comfyui': {'api_url': 'http://127.0.0.1:9'}})
        seen = {}

        def _get(url, **kw):
            seen['url'] = url
            return _Resp()
        monkeypatch.setattr(capabilities.requests, 'get', _get)
        assert capabilities.detect_comfyui_folders() == {'output_dir': r'D:\out'}
        assert seen['url'].endswith('/system_stats')


# --- the route the Settings fields call -------------------------------------------

def test_route_previews_unsaved_values(client, tmp_path):
    base = tmp_path / 'Comfy'
    (base / 'input').mkdir(parents=True)
    r = client.get(f'/api/setup/comfyui-folders?base_dir={base}')
    assert r.status_code == 200
    body = r.get_json()
    assert body['folders']['input_dir']['resolved'] == str(base / 'input')
    assert body['folders']['input_dir']['exists'] is True
    assert body['folders']['output_dir']['exists'] is False
    assert body['detected'] == {}          # detection is opt-in (?detect=1)


def test_route_reports_a_missing_override(client, tmp_path):
    ghost = tmp_path / 'ghost'
    r = client.get(f'/api/setup/comfyui-folders?base_dir={tmp_path}&output_dir={ghost}')
    info = r.get_json()['folders']['output_dir']
    assert info['source'] == 'override' and info['exists'] is False


def test_route_does_not_touch_saved_config(client, tmp_path):
    import app.config as config
    before = config.get('comfyui.output_dir')
    client.get(f'/api/setup/comfyui-folders?base_dir={tmp_path}&output_dir={tmp_path / "x"}')
    assert config.get('comfyui.output_dir') == before
