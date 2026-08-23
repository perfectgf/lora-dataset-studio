"""The Klein model used by ✨ Upscale & improve is a CHOICE, and it is the dataset's.

Reported by the maintainer: "I have no option anywhere to choose the Klein model
used for improve and upgrade". Both halves were true. `improve_existing_image`
took no model parameter, so `enqueue_klein_edit` always received `klein_model=None`
and fell back to whatever `resolve_klein_unet(None)` happened to find — while the
GENERATION lane had had a picker for months, whose value was persisted per BROWSER
(`editPage_flux2KleinModel_v1`) even though it describes what the dataset contains.

What this file pins:

* the stored choice reaches `enqueue_klein_edit` on all THREE improve lanes
  (single, re-improve, batch) — a choice honoured by one of them would be worse
  than no choice at all;
* a dataset that never chose still resolves EXACTLY the model it resolved before
  (the anti-regression that makes this migration a no-op);
* a stored model that has since left the disk is refused BY NAME instead of being
  silently swapped for another one — the old `resolve_klein_unet(selected)`
  fallback chain did the swap without a word;
* every on-disk layout the resolver accepts is also OFFERED by the picker, layout
  by layout (the list is imported from test_klein_model_locations_documented so
  the two can never drift apart).
"""
import io
import os
import struct

import pytest
from PIL import Image

from test_klein_model_locations_documented import DOCUMENTED_LAYOUTS, KLEIN_FILE

_VALID_ST = struct.pack('<Q', 2) + b'{}'
OTHER_FILE = 'flux-2-klein-32b-heavy.safetensors'


def _png():
    buf = io.BytesIO()
    Image.new('RGB', (96, 64), (25, 50, 75)).save(buf, 'PNG')
    return buf.getvalue()


def _dataset_with_source(svc, image_cls, user_id, filename='source.png'):
    ds = svc.create_dataset(user_id, 'Improve model', 'improvemodel')
    os.makedirs(svc._dataset_dir(ds.id), exist_ok=True)
    with open(os.path.join(svc._dataset_dir(ds.id), filename), 'wb') as fh:
        fh.write(_png())
    image = image_cls(dataset_id=ds.id, filename=filename, source='import',
                      status='keep', framing='body', caption='a caption',
                      variation_label='Imported', variation_prompt='p')
    svc.db.session.add(image)
    svc.db.session.commit()
    return ds, image


@pytest.fixture()
def lanes(app, monkeypatch):
    """The three improve lanes, each returning the kwargs enqueue_klein_edit saw."""
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper as keh

    queued = []
    monkeypatch.setattr(keh, 'klein_missing_assets', lambda: [])
    monkeypatch.setattr(keh, 'klein_missing_nodes', lambda: [])
    monkeypatch.setattr(svc, '_sync_generate_activity', lambda *a, **k: None)
    counter = {'n': 0}

    def _enqueue(**kwargs):
        queued.append(kwargs)
        counter['n'] += 1
        return f'job-{counter["n"]}'

    monkeypatch.setattr(keh, 'enqueue_klein_edit', _enqueue)

    class Lanes:
        calls = queued

        @staticmethod
        def single(stored=None):
            ds, src = _dataset_with_source(svc, FaceDatasetImage, LOCAL_USER)
            if stored is not None:
                svc.set_dataset_klein_model(LOCAL_USER, ds.id, stored)
            svc.improve_existing_image(LOCAL_USER, src.id)
            return ds, src

        @staticmethod
        def reimprove(stored=None):
            ds, src = _dataset_with_source(svc, FaceDatasetImage, LOCAL_USER)
            if stored is not None:
                svc.set_dataset_klein_model(LOCAL_USER, ds.id, stored)
            res = svc.improve_existing_image(LOCAL_USER, src.id)
            candidate = FaceDatasetImage.query.get(res['candidate_id'])
            # A re-improve only runs on a FINISHED candidate.
            candidate.filename = 'improved.png'
            with open(os.path.join(svc._dataset_dir(ds.id), 'improved.png'), 'wb') as fh:
                fh.write(_png())
            candidate.status = 'keep'
            svc.db.session.commit()
            queued.clear()
            svc.reimprove_image(LOCAL_USER, candidate.id)
            return ds, candidate

        @staticmethod
        def batch(stored=None):
            ds, src = _dataset_with_source(svc, FaceDatasetImage, LOCAL_USER)
            if stored is not None:
                svc.set_dataset_klein_model(LOCAL_USER, ds.id, stored)
            svc.start_bulk_improve(app, LOCAL_USER, ds.id, [src.id])
            return ds, src

    with app.app_context():
        yield Lanes


# --- 1. The choice reaches the queue, on all three lanes ---------------------
@pytest.mark.parametrize('lane', ['single', 'reimprove', 'batch'])
def test_the_stored_choice_reaches_enqueue_on_every_lane(lanes, lane):
    getattr(lanes, lane)(stored=OTHER_FILE)
    assert lanes.calls, f'{lane} enqueued nothing'
    assert lanes.calls[-1].get('klein_model') == OTHER_FILE, lane


# --- 2. Anti-regression: a dataset that never chose is untouched -------------
@pytest.mark.parametrize('lane', ['single', 'reimprove', 'batch'])
def test_a_dataset_that_never_chose_keeps_todays_model(lanes, lane):
    """`klein_model=None` is what every improve sent before this feature existed,
    and it is what makes resolve_klein_unet pick the canonical download. Anything
    else here means the migration changed the output of an untouched install."""
    getattr(lanes, lane)()
    assert lanes.calls[-1].get('klein_model') is None, lane


# --- 3. A model that left the disk is refused BY NAME -----------------------
def test_a_vanished_model_is_refused_by_name_not_swapped(app, tmp_path):
    """The old chain resolved an unknown pick to the canonical file and ran the
    job on it. A user who chose a 32B model and got a 9B result had no way to
    know: the tile looks fine, it is just not what was asked for."""
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    with app.app_context():
        base = tmp_path / 'ComfyUI'
        (base / 'models' / 'unet' / 'klein').mkdir(parents=True)
        (base / 'input').mkdir(parents=True, exist_ok=True)
        (base / 'main.py').write_text('# fake', encoding='utf-8')
        (base / 'models' / 'unet' / 'klein' / KLEIN_FILE).write_bytes(_VALID_ST)
        cfg.save_config({'comfyui': {'base_dir': str(base)}})

        # The present file still resolves, prefix and all.
        assert keh.klein_model_on_disk(KLEIN_FILE) == os.path.join('klein', KLEIN_FILE)
        # The absent one resolves to NOTHING — never to the neighbour.
        assert keh.klein_model_on_disk(OTHER_FILE) is None

        src = tmp_path / 'src.png'
        src.write_bytes(_png())
        with pytest.raises(keh.KleinModelGone) as exc:
            keh.enqueue_klein_edit(user_id='local', source_filename='src.png',
                                   source_path=str(src), edit_prompt='improve',
                                   klein_model=OTHER_FILE)
        assert OTHER_FILE in str(exc.value)
        assert exc.value.name == OTHER_FILE


# --- 4. The offered list is as WIDE as the resolver -------------------------
# Offering fewer places than the resolver accepts is the same bug seen from the
# other end: the app would refuse to let you pick a model it can happily load.
@pytest.mark.parametrize('label,parts,expected', DOCUMENTED_LAYOUTS,
                         ids=[c[0] for c in DOCUMENTED_LAYOUTS])
def test_every_resolvable_layout_is_also_offerable(app, tmp_path, label, parts, expected):
    from app import capabilities, config as cfg
    from app.services import comfy_model_paths as cmp
    from app.services import klein_edit_helper as keh
    cmp.clear_cache()
    with app.app_context():
        base = tmp_path / 'ComfyUI'
        for sub in ('input', 'output', 'models'):
            (base / sub).mkdir(parents=True, exist_ok=True)
        (base / 'main.py').write_text('# fake', encoding='utf-8')
        target = base.joinpath('models', *parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_VALID_ST)
        cfg.save_config({'comfyui': {'base_dir': str(base)}})
        offered = capabilities._scan_models()['klein']
        # Offered under the resolver's own relative name (the probe delegates
        # to the resolvers' scan since the fifth scanner folded)…
        assert expected in offered, label
        # …and picking exactly what was offered lands on exactly what the
        # resolver would have chosen by itself. A bare name still resolves too
        # (the deep-search wave's own tests pin that).
        assert keh.klein_model_on_disk(expected) == expected, label
        assert keh.klein_model_on_disk(KLEIN_FILE) == expected, label
        assert keh.resolve_klein_unet() == expected, label
    cmp.clear_cache()


# --- 5. Storing the choice -------------------------------------------------
def test_the_choice_is_stored_on_the_dataset_not_the_browser(app):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Storage', 'storage')
        assert svc.dataset_klein_model(ds) is None
        svc.set_dataset_klein_model(LOCAL_USER, ds.id, OTHER_FILE)
        assert svc.dataset_klein_model(ds) == OTHER_FILE
        # Clearing goes back to "auto" — the same NULL a fresh dataset has, so
        # un-choosing is a real gesture and not a stuck value.
        svc.set_dataset_klein_model(LOCAL_USER, ds.id, '')
        assert svc.dataset_klein_model(ds) is None


def test_a_path_separator_cannot_be_smuggled_into_the_choice(app):
    """The picker sends a BARE file name; the prefix is the resolver's job. A
    value carrying a separator would be a traversal attempt, not a model.

    BOTH separators must be refused on BOTH hosts. The guard used to be written
    with `os.path.basename`, which reads a backslash as an ordinary filename
    character on Linux — so `sub\\model.safetensors` was refused on Windows and
    accepted on every Linux/Docker install (reported by socrasteeze, GitHub #20).
    That is why these cases are asserted unconditionally rather than per-OS.
    """
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Traversal', 'traversal')
        for bad in ('../secret.safetensors', '..\\secret.safetensors',
                    'sub/model.safetensors', 'sub\\model.safetensors',
                    '/etc/passwd', 'C:\\models\\x.safetensors', '.', '..'):
            with pytest.raises(ValueError):
                svc.set_dataset_klein_model(LOCAL_USER, ds.id, bad)
        assert svc.dataset_klein_model(ds) is None
