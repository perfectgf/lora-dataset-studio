"""Importing a LoRA the user already has (2026-09-01).

The picker listed what this app trained and what was already in ComfyUI's `h3`
folder — so a LoRA downloaded from anywhere else had to be moved there by hand,
in a file explorer, with the app open beside it. This is that copy, done by the
app, with the refusals that keep the folder honest.
"""
import os

import pytest

from app.services import video_test_studio as vts


@pytest.fixture
def loras_dir(tmp_path, monkeypatch):
    dest = tmp_path / 'loras' / 'h3' / 'lds'
    dest.mkdir(parents=True)
    monkeypatch.setattr(vts, '_loras_write_dir', lambda: str(dest))
    return dest


def _lora(tmp_path, name='mine.safetensors', size=2048):
    p = tmp_path / name
    p.write_bytes(b'\x00' * size)
    return p


def test_a_lora_on_this_machine_is_copied_and_named_for_the_loader(loras_dir, tmp_path):
    src = _lora(tmp_path)
    out = vts.import_external_lora(src_path=str(src))
    assert out['filename'] == os.path.join(vts.LORA_SUBDIR, 'mine.safetensors')
    assert out['label'] == 'mine'
    assert out['already'] is False
    assert (loras_dir / 'mine.safetensors').is_file()
    # And it is then a LoRA the picker lists, like any other in that folder.
    assert out['bytes'] == src.stat().st_size


def test_re_importing_the_same_file_costs_nothing(loras_dir, tmp_path):
    src = _lora(tmp_path)
    vts.import_external_lora(src_path=str(src))
    again = vts.import_external_lora(src_path=str(src))
    assert again['already'] is True


def test_a_different_file_under_the_same_name_is_refused_never_overwritten(
        loras_dir, tmp_path):
    """Overwriting would silently change what every clip generated with that
    name meant, and the picker would show one label for two different weights."""
    vts.import_external_lora(src_path=str(_lora(tmp_path)))
    other = tmp_path / 'other'
    other.mkdir()
    bigger = _lora(other, size=4096)
    with pytest.raises(ValueError, match='different'):
        vts.import_external_lora(src_path=str(bigger))
    # The one that was there is untouched.
    assert (loras_dir / 'mine.safetensors').stat().st_size == 2048


def test_only_safetensors_and_only_names_that_stay_in_the_folder(loras_dir, tmp_path):
    ckpt = tmp_path / 'weights.ckpt'
    ckpt.write_bytes(b'\x00' * 16)
    with pytest.raises(ValueError, match='safetensors'):
        vts.import_external_lora(src_path=str(ckpt))
    # Traversal, drive letters and rooted names cannot reach outside the
    # folder: the destination is built from the BASENAME, so each of these
    # lands beside the others or is refused — never one directory up. The
    # property is what matters, not which of the two guards catches it.
    for bad in ('../escape.safetensors', 'C:/x.safetensors', '/rooted.safetensors',
                r'..\win.safetensors'):
        src = tmp_path / 'src'
        src.mkdir(exist_ok=True)
        f = _lora(src, name=os.path.basename(bad.replace(chr(92), '/')))
        try:
            out = vts.import_external_lora(src_path=str(f), filename=bad)
        except ValueError:
            continue                      # refused outright: also correct
        written = loras_dir / os.path.basename(out['filename'])
        assert written.is_file(), bad
        assert loras_dir.resolve() in written.resolve().parents, bad


def test_a_missing_file_says_so_instead_of_creating_an_empty_one(loras_dir, tmp_path):
    with pytest.raises(ValueError, match='not on this machine'):
        vts.import_external_lora(src_path=str(tmp_path / 'nope.safetensors'))
    assert not any(loras_dir.iterdir())
