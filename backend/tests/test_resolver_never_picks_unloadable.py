"""A base-model resolver must never CHOOSE a file the loader cannot open.

Reported by naniii2352 (Discord) as "it generated half the images and then
started throwing a gguf error" — with no setting touched between the two. The
cause was not his config: `resolve_krea_unet` picks the first candidate whose
NAME contains 'turbo', and `krea2_turbo-Q4_K_M.gguf` contains 'turbo'. Dropping
that file into a krea folder mid-session was enough for the app to start
choosing, on its own, a model core ComfyUI never even scans — which is why
copying it into every models directory changed nothing.

The listing side deliberately still shows .gguf (a file you can see is a
nameable problem; a file that silently vanished is not). The CHOOSING side must
not. These tests pin that difference, because it is the kind of distinction that
gets flattened by the next person who unifies two extension tuples.
"""

from app.services import comfy_model_paths as cmp


def test_gguf_is_listable_but_not_loadable():
    """The whole bug in one assertion."""
    assert '.gguf' in cmp._MODEL_EXTENSIONS, 'still listed, on purpose'
    assert '.gguf' not in cmp.LOADABLE_MODEL_EXTENSIONS
    assert cmp.is_loadable_model('krea2_turbo_fp8.safetensors') is True
    assert cmp.is_loadable_model('krea2_turbo-Q4_K_M.gguf') is False
    assert cmp.is_loadable_model('MODEL.SafeTensors') is True     # case-insensitive
    assert cmp.is_loadable_model(None) is False


def test_krea_automatic_pick_skips_a_gguf_that_matches_the_token(monkeypatch):
    """His exact folder: a .gguf named …turbo… next to a real fp8 build.

    Before the fix the .gguf won on the 'turbo' token and the job died in
    ComfyUI validation.
    """
    from app.services import krea_edit_helper as keh
    monkeypatch.setattr(keh, '_krea_unet_folders', lambda: [
        ('Krea', ['krea2_turbo-Q4_K_M.gguf', 'krea2_turbo_fp8.safetensors'])])
    monkeypatch.setattr(keh.cfg, 'get', lambda *a, **k: '')
    import os
    assert keh.resolve_krea_unet() == os.path.join('Krea', 'krea2_turbo_fp8.safetensors')


def test_krea_returns_none_when_only_unloadable_files_exist(monkeypatch):
    """No silent fallback to a file that cannot load: the caller must be able to
    say 'nothing usable here' rather than fail later inside ComfyUI."""
    from app.services import krea_edit_helper as keh
    monkeypatch.setattr(keh, '_krea_unet_folders', lambda: [('Krea', ['a.gguf', 'b.gguf'])])
    monkeypatch.setattr(keh.cfg, 'get', lambda *a, **k: '')
    assert keh.resolve_krea_unet() is None


def test_an_explicit_pick_is_still_honoured_even_if_unloadable(monkeypatch):
    """Deliberate: if the user NAMES a file, the error must be about THEIR file.

    Silently substituting another model would render something they did not ask
    for and hide the real problem — the opposite of the diagnostic we shipped.
    """
    from app.services import krea_edit_helper as keh
    monkeypatch.setattr(keh, '_krea_unet_folders', lambda: [
        ('Krea', ['krea2_turbo-Q4_K_M.gguf', 'krea2_turbo_fp8.safetensors'])])
    monkeypatch.setattr(keh.cfg, 'get', lambda *a, **k: 'krea2_turbo-Q4_K_M.gguf')
    import os
    assert keh.resolve_krea_unet() == os.path.join('Krea', 'krea2_turbo-Q4_K_M.gguf')


def test_klein_last_resort_pick_skips_unloadable(monkeypatch):
    """Klein's final fallback took names[0] — a .gguf sorting first would win."""
    from app.services import klein_edit_helper as keh
    monkeypatch.setattr(keh, '_klein_unet_folders', lambda: [
        ('klein', ['aaa_model.gguf', 'zzz_model.safetensors'])])
    monkeypatch.setattr(keh, '_canonical_name', lambda *a, **k: 'nope.safetensors')
    import os
    assert keh.resolve_klein_unet() == os.path.join('klein', 'zzz_model.safetensors')
