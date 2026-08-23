"""The four family scanners, pinned file by file on ONE tree.

WHY THIS FILE EXISTS
--------------------
Four functions walk the same ComfyUI folders looking for one family's weights:
`comfyui.get_krea_models`, `comfyui.get_zimage_models`,
`krea_edit_helper._krea_unet_folders` and `klein_edit_helper._klein_unet_folders`.
They were four hand-written walks, and their drift has already shipped a bug —
two of them answered differently about the SAME folder, so a file the picker
offered was a file the resolver refused.

These tests are a GOLDEN RECORD taken before the four walks were folded onto two
shared ones. Every list below was produced by the code as it stood, on the tree
built by the `tree` fixture; the refactor is only allowed if they all stay green.
They are deliberately exact-equality assertions on full lists rather than `in`
checks: a consolidation that quietly widens or narrows a list is exactly the
failure mode, and a membership test would sail straight past it.

WHY THE DEPTH PINS CHANGED ON 2026-08-04, AND WHAT MEASURED IT
--------------------------------------------------------------
This file used to pin a difference instead of fixing it: `get_*_models` walked
the tree, `_*_unet_folders` read ONE level, so a model two folders deep was
offered by the Test Studio and invisible to Generate. It was left standing
because which of the two matched ComfyUI was a question nobody had answered, and
levelling blind could as easily have removed working entries as added missing
ones.

It has now been measured against the ComfyUI the app actually drives (0.30.1),
and the LISTERS were right:

  * `folder_paths.get_filename_list_` calls `recursive_search(x,
    excluded_dir_names=[".git"])`, whose body is a plain
    `os.walk(directory, followlinks=True, topdown=True)` — no depth limit;
  * the name it publishes is `os.path.relpath(file, directory)`, i.e. WITH the
    subfolder, which is the same string `get_full_path` joins back onto the root
    to open the file. A deep model is therefore listed AND loadable.

Verified by experiment, not only by reading, on a running instance: fake weights
planted at depth 2 and depth 4 under `models/unet` both appeared in
`GET /object_info` under their full relative name, and a dot-directory that is
not `.git` was listed too.

  METHOD NOTE — the fourth probe is what makes the other three conclusive. A
  same-shaped file was planted under `models/unet/.git/` and stayed ABSENT while
  the others appeared. Without that negative control, three positives are also
  what a stale cache listing everything would produce; the control proves the
  list genuinely refreshed AND that absence is detectable. Any future re-measure
  of this behaviour should carry one.

So the resolvers now walk to any depth, `.git` is excluded on both sides, and the
pins below moved accordingly. The one thing NOT levelled is the position of the
root entry within its root — see `scan_family_folders`, which explains why moving
it would silently re-elect a different default base.

A second difference was levelled earlier, in its own commit and on purpose — see
`test_both_listers_offer_every_krea_file_the_user_has` below.
"""
import importlib
import os

import pytest


@pytest.fixture
def tree(monkeypatch, tmp_path):
    """One ComfyUI tree, seen by BOTH root-discovery paths: `comfyui._out_dir()`
    (which reads `<output>/../models`) and `comfy_model_paths.search_roots`
    (which reads `comfyui.base_dir`). They are two different config routes to the
    same folders — a test that wired only one would compare scanners that were
    not even looking at the same disk."""
    monkeypatch.setenv('LDS_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('LDS_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setenv('LDS_ENV', str(tmp_path / '.env'))
    import app.config as config
    importlib.reload(config)
    base = tmp_path / 'Comfy'
    (base / 'output').mkdir(parents=True)
    models = base / 'models'
    for sub in ('unet', 'diffusion_models', 'loras', 'text_encoders', 'vae'):
        (models / sub).mkdir(parents=True)
    config.save_config({'comfyui': {'base_dir': str(base)}})

    def put(*parts):
        p = models.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'W')

    # --- unet/ ---------------------------------------------------------------
    put('unet', 'Krea', 'krea2_turbo_fp8.safetensors')
    put('unet', 'Krea', 'BigLoveKreaEdit1_fp8mixed.safetensors')   # noise base
    put('unet', 'Krea', 'notes.txt')                               # not a model
    put('unet', 'Krea', 'nested', 'deep_krea.safetensors')         # depth 2
    put('unet', 'Krea', 'quant.gguf')
    put('unet', 'BigLoveKreaEdit1_fp8mixed.safetensors')           # noise, at root
    put('unet', 'Krea_full_x_fp8.safetensors')                     # root, named krea
    put('unet', 'flux2-klein-9b.safetensors')                      # another family
    put('unet', 'Krea', '.git', 'ghost_krea.safetensors')          # ComfyUI skips .git
    put('unet', 'z image', 'zt3.safetensors')
    put('unet', 'z image', 'sub', 'deep_z.safetensors')            # depth 2
    put('unet', 'klein', 'flux2-klein-4b.safetensors')
    put('unet', 'klein', 'deep', 'deep_klein.safetensors')         # depth 2
    # --- diffusion_models/ ---------------------------------------------------
    put('diffusion_models', 'krea2', 'another_krea.safetensors')
    put('diffusion_models', 'zimage', 'z_merge.sft')
    put('diffusion_models', 'Flux2 klein', 'klein9b.safetensors')
    put('diffusion_models', 'klein_root.safetensors')

    from app.services import comfy_model_paths
    from app.utils import comfyui
    comfy_model_paths.clear_cache()
    monkeypatch.setattr(comfyui, '_out_dir', lambda: str(base / 'output'))
    comfyui.clear_model_caches()
    yield base
    comfy_model_paths.clear_cache()
    comfyui.clear_model_caches()


def _j(*parts):
    return os.path.join(*parts)


# --- Krea, the Studio's list --------------------------------------------------

def test_get_krea_models_is_pinned_file_by_file(tree):
    from app.utils.comfyui import get_krea_models
    assert get_krea_models() == [
        'BigLoveKreaEdit1_fp8mixed.safetensors',
        _j('Krea', 'BigLoveKreaEdit1_fp8mixed.safetensors'),
        _j('Krea', 'krea2_turbo_fp8.safetensors'),
        _j('Krea', 'nested', 'deep_krea.safetensors'),
        _j('Krea', 'quant.gguf'),
        'Krea_full_x_fp8.safetensors',
        _j('krea2', 'another_krea.safetensors'),
    ]


def test_both_listers_offer_every_krea_file_the_user_has(tree):
    """THE BEHAVIOUR CHANGE, and the reason for it.

    `BigLove*` carries 'krea' and renders PURE NOISE under the Krea pipeline —
    measured. That fact used to REMOVE it from the lists, and the two surfaces did
    not even agree about where: the Studio dropped it at a root only, Generate
    everywhere, so the same file was offered or hidden depending on which folder
    it happened to sit in.

    Both now list it. Hiding a file that sits on the user's own disk told them
    nothing — not that it existed, not why it was gone — and choosing is theirs to
    do. The measured fact did not disappear: it moved from filter to warning, and
    to `elect_krea_base`, which still refuses to PREFER a flagged build when the
    app is the one choosing (test_krea_default_base_election)."""
    from app.utils.comfyui import get_krea_models
    from app.services import krea_edit_helper as keh
    importlib.reload(keh)
    listed = get_krea_models()
    resolver = [n for _sub, group in keh._krea_unet_folders() for n in group]
    assert any('biglove' in m.lower() for m in listed), (
        'the Studio hides a file the user put in their own Krea folder')
    assert any('biglove' in n.lower() for n in resolver), (
        'Generate hides a file the user put in their own Krea folder')


# --- Z-Image, the Studio's list ----------------------------------------------

def test_get_zimage_models_is_pinned_file_by_file(tree):
    from app.utils.comfyui import get_zimage_models
    assert get_zimage_models() == [
        _j('z image', 'sub', 'deep_z.safetensors'),
        _j('z image', 'zt3.safetensors'),
        _j('zimage', 'z_merge.sft'),
    ]


def test_a_zimage_file_at_a_root_is_not_listed(tree):
    """Unlike Krea, this family has no root-filename rule: the folder is the only
    claim. Pinned because the two listers look alike and are not."""
    from app.utils.comfyui import get_zimage_models
    assert not any(os.sep not in m for m in get_zimage_models())


# --- Krea, the Generate resolver's folders ------------------------------------

def test_krea_unet_folders_is_pinned_group_by_group(tree):
    from app.services import krea_edit_helper as keh
    importlib.reload(keh)
    assert keh._krea_unet_folders() == [
        ('Krea', ['BigLoveKreaEdit1_fp8mixed.safetensors',
                  'krea2_turbo_fp8.safetensors', 'quant.gguf']),
        (_j('Krea', 'nested'), ['deep_krea.safetensors']),
        ('', ['BigLoveKreaEdit1_fp8mixed.safetensors',
              'Krea_full_x_fp8.safetensors']),
        ('krea2', ['another_krea.safetensors']),
    ]


def test_the_resolver_lists_the_noise_base_at_every_depth(tree):
    """The other half of the change, on the same tree and the same file: a
    subfolder is no longer a place where a file quietly disappears either."""
    from app.services import krea_edit_helper as keh
    importlib.reload(keh)
    names = [n for _sub, group in keh._krea_unet_folders() for n in group]
    assert any('biglove' in n.lower() for n in names)


def test_the_resolver_sees_a_model_two_folders_deep_like_the_studio_does(tree):
    """THE DEPTH FIX, from both ends. Was `..._never_sees_...`, asserting the
    opposite: the same file, offered by the Studio and invisible to Generate.

    Measured on ComfyUI 0.30.1 (see this module's header): a model at depth 2 is
    listed under its full relative name and loads from it. So the resolver was
    the half that was wrong, and levelling means the deep file becomes REACHABLE
    — no entry the app used to offer is taken away.

    Both ends asserted on purpose. A version of this that only checked the
    resolver would still pass if the lister had been narrowed to match instead,
    which is the opposite fix and the one that would have cost users entries."""
    from app.services import krea_edit_helper as keh
    from app.utils.comfyui import get_krea_models
    importlib.reload(keh)
    resolver = [_j(sub, n) for sub, group in keh._krea_unet_folders() for n in group]
    assert _j('Krea', 'nested', 'deep_krea.safetensors') in resolver
    assert _j('Krea', 'nested', 'deep_krea.safetensors') in get_krea_models()


def test_the_deep_model_resolves_to_the_value_a_loader_can_open(tree):
    """The picker stores a BASENAME, so reachability is not enough: the resolver
    has to hand back the value WITH its subfolder, or the job loads nothing.

    This is the user-visible half of the Krea bug. `resolve_krea_unet` matches the
    pick on its basename across the family's folders; at one level it found
    nothing, logged "not found under any krea folder" and fell through to
    `elect_krea_base` — so the job silently ran on a DIFFERENT base than the one
    chosen, and the output looked plausible enough never to be questioned."""
    from app.services import krea_edit_helper as keh
    importlib.reload(keh)
    assert keh.resolve_krea_unet('deep_krea.safetensors') == _j(
        'Krea', 'nested', 'deep_krea.safetensors')


def test_a_shallower_file_still_wins_a_basename_collision(tree, monkeypatch,
                                                          tmp_path):
    """The tie-break, pinned. Two files share a basename at different depths; the
    resolver must return the SHALLOWER one — `resolve_model_file`'s existing rule
    (shortest relative path, then alphabetical), reused rather than reinvented.

    Two tie-break rules would only ever disagree on a collision, which is to say
    rarely, which is to say long after whoever wrote the second one had moved on.
    That is precisely how four hand-written scanners drifted into a shipped bug."""
    from app.services import krea_edit_helper as keh
    importlib.reload(keh)
    twin = tmp_path / 'Comfy' / 'models' / 'unet' / 'Krea' / 'nested' / 'twin.safetensors'
    twin.write_bytes(b'W')
    (twin.parent.parent / 'twin.safetensors').write_bytes(b'W')
    assert keh.resolve_krea_unet('twin.safetensors') == _j('Krea', 'twin.safetensors')


def test_neither_half_offers_a_model_comfyui_hides_in_dot_git(tree):
    """ABSENT where ComfyUI does not look. `recursive_search` excludes exactly one
    directory name, `.git`, so a model under it is never listed — and ComfyUI
    validates a loader's combo value against that list, so a job naming one would
    be refused at queue time. Offering it would be the `.gguf` trap again: a file
    the app shows and the engine will not take.

    Measured, not assumed — this was the negative control of the live probe."""
    from app.services import krea_edit_helper as keh
    from app.utils.comfyui import get_krea_models
    importlib.reload(keh)
    resolver = [n for _sub, group in keh._krea_unet_folders() for n in group]
    assert 'ghost_krea.safetensors' not in resolver
    assert not any('ghost_krea' in m for m in get_krea_models())


# --- Klein, the Generate resolver's folders -----------------------------------

def test_klein_unet_folders_is_pinned_group_by_group(tree):
    from app.services import klein_edit_helper as kleh
    importlib.reload(kleh)
    assert kleh._klein_unet_folders() == [
        ('klein', ['flux2-klein-4b.safetensors']),
        (_j('klein', 'deep'), ['deep_klein.safetensors']),
        ('', ['flux2-klein-9b.safetensors']),
        ('Flux2 klein', ['klein9b.safetensors']),
        ('', ['klein_root.safetensors']),
    ]


def test_klein_keeps_every_candidate_it_finds(tree):
    """Klein has no incompatible-base list — pinned so a consolidation cannot
    hand it Krea's exclusion by accident."""
    from app.services import klein_edit_helper as kleh
    importlib.reload(kleh)
    groups = kleh._klein_unet_folders()
    assert sum(len(names) for _sub, names in groups) == 5


def test_klein_can_name_every_model_the_ui_offers(tree):
    """The invariant `klein_model_on_disk` DOCUMENTS — "so a model the UI offers
    is always one this can name" — asserted instead of merely written down.

    It was written and not held: the docstring promised it while the function
    scanned ONE level, so a model two folders down returned None, which the caller
    turns into `KleinModelGone` — "the Klein model chosen for this dataset is no
    longer on disk" — about a file that is on disk the whole time.

    Honest about the blast radius: that was LATENT, not something users hit. The
    Klein picker's choices come from `capabilities._scan_models`, a third scanner
    still reading one level, so a deep Klein model is not offered in the UI today
    and the contradiction stayed theoretical. It is asserted anyway, because an
    invariant stated in prose and contradicted by the code beneath it is worth
    less than no invariant at all — it stops the next reader from checking. This
    test is also what will FAIL, loudly and in the right file, on the day
    `_scan_models` is widened to match."""
    from app.services import klein_edit_helper as kleh
    importlib.reload(kleh)
    offered = [_j(sub, n) for sub, names in kleh._klein_unet_folders()
               for n in names]
    assert _j('klein', 'deep', 'deep_klein.safetensors') in offered
    for value in offered:
        assert kleh.klein_model_on_disk(os.path.basename(value)) is not None
