"""Every deployable family must be testable, and must read its OWN folder.

GitHub #52 (lunchingfriar): a Klein LoRA trained in the cloud deployed fine,
Undeploy listed it with the right folder and file name, and every "is it
deployed?" answer was still no. Deploy wrote `loras/flux2klein`; the pool
dispatcher had no branch for that family and fell through to `loras/z image`,
where the file will never be. The badge never flipped, and Generate refused a
checkpoint sitting on disk. FLUX.1 and Anima were in the same hole, silently.

The bug was not a missing `if`. It was THREE tables describing the same set of
families, drifting apart with nothing to notice:

  * lora_training._FAMILY_SUBDIR   which folder each family deploys to
  * face_dataset_service.TRAIN_TYPES  which family a dataset may be trained as
  * lora_test_studio.FAMILIES      which families the studio can see

Deploy support for FLUX.2 Klein landed 2026-07-29 and the first two tables were
updated; the third was not, and nothing failed for a month. These tests are the
thing that would have failed on that commit.
"""
from app.services import lora_test_studio as studio
from app.services import lora_training as lt
from app.services import face_dataset_service as fds


def test_every_deployable_family_is_a_testable_family():
    """A family the app can deploy TO must be a family the app can look IN."""
    deployable = set(lt._FAMILY_SUBDIR)
    testable = set(studio.FAMILIES)
    missing = sorted(deployable - testable)
    assert not missing, (
        f'{missing} can be deployed but not seen: lora_test_studio.FAMILIES is '
        'missing them, so their LoRAs read as never deployed and they never '
        'appear in the Test Studio family picker (GitHub #52)')


def test_the_three_family_tables_agree():
    """No family may exist in one table and not the others."""
    assert set(studio.FAMILIES) == set(lt._FAMILY_SUBDIR) == set(fds.TRAIN_TYPES)


def test_every_family_reads_a_pool_of_its_own(monkeypatch):
    """The failure mode was a SHARED pool, not an empty one: Klein quietly got
    Z-Image's folder, so its answers were wrong rather than absent. Pin that each
    family resolves to its own distinct folder."""
    seen = {}

    def fake_dirs(fam):
        return [f'/loras/{lt._FAMILY_SUBDIR[fam]}']
    monkeypatch.setattr(lt, '_lora_family_dirs', fake_dirs)

    for fam in studio.FAMILIES:
        folder = lt._FAMILY_SUBDIR[fam]
        assert folder not in seen, (
            f'{fam} and {seen.get(folder)} both read {folder!r}: one of them is '
            'looking in the other one\'s folder')
        seen[folder] = fam


def test_an_unknown_family_reads_nothing_rather_than_z_image():
    """The silent fallback IS the bug. An unknown family used to be handed the
    Z-Image pool, so a wrong folder read exactly like an empty one and nothing
    said so. It must now answer "no LoRA" honestly."""
    assert studio._pool_for_family('not-a-family') == []


def test_the_default_family_is_still_z_image():
    """...but no family at all keeps meaning Z-Image, which is what every caller
    that omits it has always relied on."""
    calls = []
    import app.utils.comfyui as cu
    original = cu.get_zimage_loras
    try:
        cu.get_zimage_loras = lambda: calls.append('zimage') or []
        studio.get_zimage_loras = cu.get_zimage_loras
        assert studio._pool_for_family(None) == []
        assert studio._pool_for_family('') == []
        assert calls == ['zimage', 'zimage']
    finally:
        cu.get_zimage_loras = original
        studio.get_zimage_loras = original
