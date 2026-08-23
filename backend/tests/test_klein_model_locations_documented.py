"""Every Klein model location the docs PROMISE must really resolve.

Why a dedicated file rather than one more case in test_klein_models.py: the
promise is made in prose (README "Where the Klein model can live",
docs/guide/troubleshooting.md, the Setup screen) and prose has no compiler.
CyberTod (Reddit) read `unet\\klein` as a requirement, duplicated ~10 GB of
weights into it and built a symlink to get the space back — for nothing, because
the resolver had already scanned five other places for two releases. A
documentation claim that nothing executes goes stale at the first refactor; this
file is the executable copy of that documentation.

Each case below is ONE line of the docs. If you delete a case, delete the line.

The scan and the resolver are asserted TOGETHER on purpose: the picker listing a
file the resolver cannot name (or the reverse) is the exact shape of the original
`klein/`-prefix bug.
"""
import os
import struct
import textwrap

import pytest

# Smallest structurally-valid safetensors container (8-byte LE length + '{}'):
# model_integrity rejects an empty stub, and the readiness gate keys off it.
_VALID_ST = struct.pack('<Q', 2) + b'{}'

# The canonical Setup download. Only its NAME matters here.
KLEIN_FILE = 'flux-2-klein-9b-kv-fp8.safetensors'


@pytest.fixture(autouse=True)
def _clear_cmp_cache():
    from app.services import comfy_model_paths as cmp
    cmp.clear_cache()
    cmp._warned.clear()
    yield
    cmp.clear_cache()
    cmp._warned.clear()


def _weights(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as fh:
        fh.write(_VALID_ST)
    return path


def _save(patch):
    from app import config as cfg
    cfg.save_config(patch)


def _base(tmp_path):
    """A minimal ComfyUI install with NO Klein model anywhere yet."""
    base = tmp_path / 'ComfyUI'
    for sub in ('input', 'output', 'models'):
        (base / sub).mkdir(parents=True, exist_ok=True)
    (base / 'main.py').write_text('# fake', encoding='utf-8')
    return base


# --- The documented layouts -------------------------------------------------
# (id, relative path under <models>/ where the file is dropped, expected
#  loader-relative `unet_name`). `expected` is what a ComfyUI UNETLoader lists,
#  i.e. the path relative to the search ROOT that holds the file.
DOCUMENTED_LAYOUTS = [
    # The Setup download's own destination — the one everybody thinks is mandatory.
    ('unet/klein subfolder', ('unet', 'klein', KLEIN_FILE), os.path.join('klein', KLEIN_FILE)),
    # Any subfolder whose NAME contains "klein", any capitalisation, spaces allowed.
    ('unet/<any klein-named subfolder>', ('unet', 'Flux2 Klein', KLEIN_FILE),
     os.path.join('Flux2 Klein', KLEIN_FILE)),
    # Loose at the root of models/unet — no subfolder at all.
    ('unet root, no subfolder', ('unet', KLEIN_FILE), KLEIN_FILE),
    # The other default diffusion-model root ComfyUI registers.
    ('diffusion_models/klein subfolder', ('diffusion_models', 'klein', KLEIN_FILE),
     os.path.join('klein', KLEIN_FILE)),
    ('diffusion_models/<any klein-named subfolder>',
     ('diffusion_models', 'flux2-klein-9b', KLEIN_FILE),
     os.path.join('flux2-klein-9b', KLEIN_FILE)),
    ('diffusion_models root, no subfolder', ('diffusion_models', KLEIN_FILE), KLEIN_FILE),
]


@pytest.mark.parametrize('label,parts,expected',
                         DOCUMENTED_LAYOUTS,
                         ids=[c[0] for c in DOCUMENTED_LAYOUTS])
def test_documented_layout_resolves(app, tmp_path, label, parts, expected):
    from app import capabilities
    from app.services import klein_edit_helper as keh
    with app.app_context():
        base = _base(tmp_path)
        _weights(str(base.joinpath('models', *parts)))
        _save({'comfyui': {'base_dir': str(base)}})
        assert keh.resolve_klein_unet() == expected, label
        # …and the picker lists it under the resolver's own name, so choosing
        # what was offered resolves identically (a bare name still works too).
        assert expected in capabilities._scan_models()['klein'], label
        assert keh.resolve_klein_unet(KLEIN_FILE) == expected, label
        # …and the generate preflight no longer calls the model missing.
        assert 'klein_model' not in keh.klein_missing_assets(), label


def test_extra_model_paths_root_resolves(app, tmp_path):
    """`extra_model_paths.yaml` roots — portable / Stability-Matrix / A1111-shared
    installs keep their weights OUTSIDE <base>/models and declare them here. Both a
    klein-named subfolder and a loose root-level file are documented."""
    from app import capabilities
    from app.services import klein_edit_helper as keh
    with app.app_context():
        base = _base(tmp_path)
        shared = tmp_path / 'shared-weights'
        _weights(str(shared / 'unet' / 'klein' / KLEIN_FILE))
        (base / 'extra_model_paths.yaml').write_text(textwrap.dedent(f"""
            my_portable_install:
              base_path: {shared}
              unet: unet
        """), encoding='utf-8')
        _save({'comfyui': {'base_dir': str(base)}})
        assert keh.resolve_klein_unet() == os.path.join('klein', KLEIN_FILE)
        assert os.path.join('klein', KLEIN_FILE) in capabilities._scan_models()['klein']
        assert 'klein_model' not in keh.klein_missing_assets()


def test_extra_model_paths_root_level_file_resolves(app, tmp_path):
    from app.services import klein_edit_helper as keh
    with app.app_context():
        base = _base(tmp_path)
        shared = tmp_path / 'shared-weights'
        _weights(str(shared / 'diffusion_models' / KLEIN_FILE))
        (base / 'extra_model_paths.yaml').write_text(textwrap.dedent(f"""
            stability_matrix:
              base_path: {shared}
              diffusion_models: diffusion_models
        """), encoding='utf-8')
        _save({'comfyui': {'base_dir': str(base)}})
        assert keh.resolve_klein_unet() == KLEIN_FILE
        assert 'klein_model' not in keh.klein_missing_assets()


def test_models_dir_override_resolves(app, tmp_path):
    """`comfyui.models_dir` (Settings ▸ Local tools ▸ ComfyUI models folder): a whole
    models/ tree that is NOT <base>/models. Documented as the answer for a ComfyUI
    launched with a relocated models directory."""
    from app import capabilities
    from app.services import klein_edit_helper as keh
    with app.app_context():
        base = _base(tmp_path)
        elsewhere = tmp_path / 'D-drive-models'
        _weights(str(elsewhere / 'unet' / 'klein' / KLEIN_FILE))
        _save({'comfyui': {'base_dir': str(base), 'models_dir': str(elsewhere)}})
        assert keh.resolve_klein_unet() == os.path.join('klein', KLEIN_FILE)
        assert os.path.join('klein', KLEIN_FILE) in capabilities._scan_models()['klein']
        assert 'klein_model' not in keh.klein_missing_assets()


def test_no_copy_and_no_symlink_needed_for_a_shared_install(app, tmp_path):
    """The whole point of CyberTod's report: a model living in ONE place outside
    unet/klein/ is enough. Nothing is copied, nothing is linked, and the canonical
    folder does not even exist."""
    from app.services import klein_edit_helper as keh
    with app.app_context():
        base = _base(tmp_path)
        _weights(str(base / 'models' / 'diffusion_models' / KLEIN_FILE))
        _save({'comfyui': {'base_dir': str(base)}})
        assert not (base / 'models' / 'unet' / 'klein').exists()
        assert keh.resolve_klein_unet() == KLEIN_FILE


# --- The documented LIMIT ---------------------------------------------------
# "Every limit stays visible": the one real constraint is that the model must be
# NAMEABLE as Klein — either the file name or its subfolder name must contain
# "klein". Documenting the freedom without documenting this is how the next
# person spends an evening wondering why their `model.safetensors` is invisible.
def test_a_root_file_must_carry_klein_in_its_name(app, tmp_path):
    from app import capabilities
    from app.services import klein_edit_helper as keh
    with app.app_context():
        base = _base(tmp_path)
        _weights(str(base / 'models' / 'unet' / 'flux2-9b-kv-fp8.safetensors'))
        _save({'comfyui': {'base_dir': str(base)}})
        assert keh.resolve_klein_unet() is None
        assert capabilities._scan_models()['klein'] == []
        assert 'klein_model' in keh.klein_missing_assets()


def test_a_subfolder_named_klein_rescues_any_filename(app, tmp_path):
    """…and the documented workaround for that limit: put it in a klein-named
    folder and the file may be called anything."""
    from app.services import klein_edit_helper as keh
    with app.app_context():
        base = _base(tmp_path)
        _weights(str(base / 'models' / 'unet' / 'klein' / 'model.safetensors'))
        _save({'comfyui': {'base_dir': str(base)}})
        assert keh.resolve_klein_unet() == os.path.join('klein', 'model.safetensors')
