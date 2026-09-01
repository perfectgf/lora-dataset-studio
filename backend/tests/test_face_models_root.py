"""antelopev2 lands where the install can keep it.

Community report (Discord): "Person grouping keeps trying to download the
antelopev2 file every time I restart the container. Did I maybe forget to mount
a path in my docker config so it's persistent or could it be a bug? The rest of
the dependencies stays installed after reboots, seems to be just this one now."

Nothing was missing from their mounts. Face work was the only engine that never
placed its own weights: with no ``root=``, insightface writes ~350 MB under
``~/.insightface``. That home is permanent on a native install, which is why the
default survived this long unseen — and in Docker it sits in the container's
writable layer, which no Compose file mounts and which ``--force-recreate`` (how
the launcher starts a STOPPED container) discards. The ML venvs live under
``data/envs``, on the mounted volume: that is exactly why "the rest of the
dependencies stays installed" and this one alone did not.

Two things are pinned here. What the resolver answers — and the fact that every
face surface hands that answer to its child. There are four producers, across
both surfaces (the Bank's face pass, its folder sample check, the dataset
scorer, the mask detector); one that forgets is the whole bug back.
"""
import json
import pathlib
import sys
from collections import deque

import pytest
from PIL import Image

from app import config as cfg
from app.services import face_models


@pytest.fixture()
def home(tmp_path, monkeypatch):
    """A home of our own. The resolver reads ``~``, and the machine running this
    suite may well hold a pack there from before this module existed — a test
    that green-lights only on a bare machine is not a test."""
    root = tmp_path / 'home' / '.insightface'
    monkeypatch.setattr(face_models, 'legacy_root', lambda: str(root))
    return root


def _install_pack(root, nested=False):
    """Put a pack where insightface would. ``nested`` is the layout its own zip
    actually produces (a root folder inside the archive)."""
    d = pathlib.Path(root) / 'models' / face_models.PACK
    if nested:
        d = d / face_models.PACK
    d.mkdir(parents=True, exist_ok=True)
    for stem in face_models.PACK_MODELS:
        (d / f'{stem}.onnx').write_bytes(b'')
    return d


def _managed(app):
    with app.app_context():
        return str(cfg.data_dir() / 'models' / 'insightface')


# --- what the resolver answers ----------------------------------------------

def test_a_configured_root_still_wins_verbatim(app, home):
    r"""Verbatim, separators included: it is the user's path. Running it through
    ``Path`` would hand the child ``C:\models`` where they typed ``C:/models``."""
    from app.config import save_config
    _install_pack(home)
    with app.app_context():
        save_config({'face_scoring': {'models_root': 'C:/weights/insightface'}})
        assert face_models.models_root() == 'C:/weights/insightface'


def test_with_nothing_installed_the_pack_lands_under_the_data_directory(app, home):
    """The fix in one assertion: an unconfigured install downloads into the one
    folder every Docker stack mounts, instead of a home nobody mounts."""
    with app.app_context():
        assert face_models.models_root() == str(cfg.data_dir() / 'models' / 'insightface')


def test_a_pack_already_sitting_in_the_home_directory_is_adopted(app, home):
    """A native install that downloaded before this module keeps its 350 MB.
    Moving those files could break another tool sharing the folder, and copying
    them would spend 350 MB fixing a path that was never broken there."""
    _install_pack(home)
    with app.app_context():
        assert face_models.models_root() == str(home)


def test_the_nested_layout_insightface_downloads_counts_as_installed(app, home):
    """antelopev2.zip carries a root folder, so a fresh auto-download lands one
    level too deep and the workers flatten it on load. Nested is downloaded."""
    _install_pack(home, nested=True)
    with app.app_context():
        assert face_models.models_root() == str(home)


def test_once_the_managed_root_holds_the_pack_the_home_copy_is_ignored(app, home):
    """Both populated is not ambiguous: ours wins, so a container that has
    already downloaded once never drifts back to the ephemeral home."""
    _install_pack(home)
    managed = _managed(app)
    _install_pack(managed)
    with app.app_context():
        assert face_models.models_root() == managed


def test_an_empty_pack_folder_is_not_a_pack(app, home):
    """insightface skips the download when the DIRECTORY exists, which is how a
    half-unzipped pack survives forever. Presence is .onnx files, not a folder."""
    (home / 'models' / face_models.PACK).mkdir(parents=True)
    with app.app_context():
        assert face_models.models_root() == str(cfg.data_dir() / 'models' / 'insightface')


def test_a_half_extracted_pack_never_outranks_a_complete_one(app, home):
    """The pack arrives as one zip of FIVE models that extractall writes one by
    one, so a run killed mid-extraction leaves a folder insightface will never
    re-download into. Counting a single .onnx would send every face pass at that
    dead root and leave a complete pack untouched next door."""
    _install_pack(home)
    managed = pathlib.Path(_managed(app))
    partial = managed / 'models' / face_models.PACK
    partial.mkdir(parents=True)
    (partial / 'genderage.onnx').write_bytes(b'')
    with app.app_context():
        assert face_models.models_root() == str(home)


# --- and every surface hands it to its child --------------------------------

def _png(color=(255, 0, 0)):
    import io
    buf = io.BytesIO()
    Image.new('RGB', (64, 64), color).save(buf, 'PNG')
    return buf.getvalue()


def _bank_with_a_folder(client, tmp_path):
    src = tmp_path / 'src'
    (src / 'anna').mkdir(parents=True)
    for i in range(3):
        Image.new('RGB', (64, 64), (10 * i, 10 * i, 10 * i)).save(
            str(src / 'anna' / f'a{i}.jpg'), 'JPEG')
    r = client.post('/api/bank/create', json={'name': 'B', 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    return r.get_json()['id']


def _fresh_job(kind):
    return {'kind': kind, 'done': 0, 'total': 0, 'error': None, 'cancelled': False,
            'finished': False, 'detail': None, '_touched': 0, '_cancel_hook': None,
            'pipeline': None}


def test_every_face_surface_hands_its_child_the_resolved_root(
        client, tmp_path, app, monkeypatch, home):
    """FIVE producers, one answer. The Bank pass is the one the report named; the
    other four would have gone on downloading into the same lost home. The count
    is the test: an adversarial pass put ``or None`` back in the folder-scan
    round and 113 tests stayed green, because nothing exercised it."""
    from app.services import face_mask, face_similarity as fsim, folder_person
    from app.services import image_bank_service as banks

    expected = _managed(app)
    seen = {}

    monkeypatch.setattr(banks, '_resolve_face_device', lambda: ('cpu', False))

    def fake_driver(job, python, script, payload, cache_path, rx, window):
        seen[job['kind']] = json.loads(payload)['models_root']
        imgs = json.loads(payload)['images']
        return ({'ok': True,
                 'results': {p: {'state': 'scorable', 'det': 0.9} for p in imgs},
                 'clusters': {p: 1 for p in imgs}}, deque(), 0)

    monkeypatch.setattr(banks, '_drive_infer_subprocess', fake_driver)

    bank_id = _bank_with_a_folder(client, tmp_path)
    with app.app_context():
        banks._faces_job(bank_id)(_fresh_job('faces'))
    client.post(f'/api/bank/{bank_id}/folder-person', json={'subfolder': 'anna'})
    with app.app_context():
        folder_person._sample_job(bank_id, 'anna')(_fresh_job('folder-check'))
        # The folder SCAN is a second, separate payload in the same module — the
        # preflight of the pass, not the sample check behind the button.
        groups = [{'name': 'anna',
                   'images': [str(tmp_path / 'src' / 'anna' / f'a{i}.jpg')
                              for i in range(3)]}]
        folder_person._embed_round(_fresh_job('folder-scan'), bank_id, groups,
                                   allow_inference=False)

    # The dataset scorer and the mask detector drive their child through a
    # different seam, so they get their own trap rather than a shared assumption.
    monkeypatch.setattr(fsim, 'is_available', lambda: True)
    monkeypatch.setattr(face_mask, 'is_available', lambda: True)

    def fake_scorer(python, payload, *a, **k):
        seen['dataset-scorer'] = json.loads(payload)['models_root']
        return (json.dumps({'ref_ok': True, 'results': {}}), [], 0, False)

    def fake_masker(python, script, payload, *a, **k):
        seen['mask-detector'] = json.loads(payload)['models_root']
        return (json.dumps({'ok': True, 'results': {}}), [], 0, False)

    monkeypatch.setattr(fsim, '_run_scorer', fake_scorer)
    monkeypatch.setattr(face_mask, 'run_infer_script', fake_masker)

    ref, img = tmp_path / 'ref.png', tmp_path / 'img.png'
    ref.write_bytes(_png())
    img.write_bytes(_png((0, 255, 0)))
    with app.app_context():
        fsim.score_dataset_faces(str(ref), [str(img)])
        face_mask.detect_faces([str(img)])

    assert set(seen) == {'faces', 'folder-check', 'folder-scan',
                         'dataset-scorer', 'mask-detector'}
    assert set(seen.values()) == {expected}


# --- the carcass an interrupted download leaves behind ------------------------

def _infer():
    """The workers run in their own interpreter and cannot import the app, so
    they carry their own copy of the pack vocabulary. Loaded the way they are."""
    sys.path.insert(0, str(cfg.BACKEND_DIR / 'infer'))
    import face_score_infer as fsi
    return fsi


def test_the_worker_and_the_app_agree_on_what_the_pack_contains():
    """Two copies of one list, by necessity (separate interpreters). Pinned here
    so they cannot drift: a worker that discards on a different list would delete
    a pack the resolver had just called complete."""
    fsi = _infer()
    assert (fsi.PACK, fsi.PACK_MODELS) == (face_models.PACK, face_models.PACK_MODELS)


def test_an_incomplete_pack_in_a_managed_root_is_discarded_so_it_downloads_again(tmp_path):
    """insightface skips the download whenever the folder exists, so without this
    the user is stuck for good — and on the mounted volume, "for good" now means
    across container recreations too."""
    fsi = _infer()
    root = tmp_path / 'data' / 'models' / 'insightface'
    outer = root / 'models' / face_models.PACK
    outer.mkdir(parents=True)
    (outer / 'genderage.onnx').write_bytes(b'')
    (root / 'models' / f'{face_models.PACK}.zip').write_bytes(b'stale')

    assert fsi._discard_incomplete_pack(str(root)) is True
    assert not outer.exists()
    assert not (root / 'models' / f'{face_models.PACK}.zip').exists()


def test_a_complete_pack_is_never_discarded(tmp_path):
    fsi = _infer()
    root = tmp_path / 'data' / 'models' / 'insightface'
    _install_pack(root)
    assert fsi._discard_incomplete_pack(str(root)) is False
    assert len(list((root / 'models' / face_models.PACK).glob('*.onnx'))) == 5


def test_insightfaces_own_default_folder_is_never_discarded(tmp_path, monkeypatch):
    """``~/.insightface`` may be shared with another tool that put those files
    there. The app clears carcasses out of roots IT chose, nothing else — and the
    rule is that exact folder, not "under the home": on Windows the install
    itself usually lives under the home, where this repair is needed most."""
    fsi = _infer()
    fake_home = tmp_path / 'home'
    monkeypatch.setenv('USERPROFILE', str(fake_home))
    monkeypatch.setenv('HOME', str(fake_home))
    root = fake_home / '.insightface'
    outer = root / 'models' / face_models.PACK
    outer.mkdir(parents=True)
    (outer / 'genderage.onnx').write_bytes(b'')

    assert fsi._discard_incomplete_pack(str(root)) is False
    assert (outer / 'genderage.onnx').exists()


def test_a_carcass_inside_the_home_is_still_discarded_when_it_is_ours(tmp_path, monkeypatch):
    """The Windows case the previous rule got wrong: data/ under the user's home
    is the normal install layout, and a carcass there must still be cleared."""
    fsi = _infer()
    monkeypatch.setenv('USERPROFILE', str(tmp_path))
    monkeypatch.setenv('HOME', str(tmp_path))
    root = tmp_path / 'lora-dataset-studio' / 'data' / 'models' / 'insightface'
    outer = root / 'models' / face_models.PACK
    outer.mkdir(parents=True)
    (outer / 'genderage.onnx').write_bytes(b'')

    assert fsi._discard_incomplete_pack(str(root)) is True
    assert not outer.exists()
