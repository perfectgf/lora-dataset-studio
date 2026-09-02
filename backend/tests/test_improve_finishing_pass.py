"""Where the finishing pass is allowed to run, and where it must not.

`photo_finish` itself is covered by test_photo_finish.py — the arithmetic. What
is pinned HERE is the wiring, and every one of these is a silent failure:

* The completion callback links EVERY dataset image job — variations, reference
  edits, the small-image rescue. A finishing pass that ran on all of them would
  quietly sharpen and grain freshly generated images that nobody asked to
  finish, and nothing would report it.
* Colour matching must not run on the SeedVR2 lane. That engine already grades
  its result back onto its source inside the node, so ours would be a second
  transform estimated from the same statistics, fighting the first.
* The whole thing is wrapped: a finishing pass may never be the reason a render
  the user waited for is lost.
"""
import json
import os

import numpy as np
import pytest

from app import config as cfg
from app.services import face_dataset_service as svc
from app.utils import photo_finish as pf


class _Img:
    """The three attributes the hook reads. A real FaceDatasetImage would drag in
    a session and a dataset row for a function that only needs an id, a parent
    and a kind."""
    def __init__(self, kind, image_id=7, parent_image_id=None):
        self.derivation_kind = kind
        self.id = image_id
        self.parent_image_id = parent_image_id


def _settings(monkeypatch, **values):
    monkeypatch.setattr(cfg, 'get',
                        lambda dotted, default=None: values.get(dotted, default))


def _write(path, seed=0):
    from PIL import Image
    rng = np.random.default_rng(seed)
    arr = np.clip(rng.normal(0.5, 0.15, (48, 48, 3)), 0, 1).astype(np.float32)
    Image.fromarray(pf.to_uint8(arr), 'RGB').save(path)


# --- Which dials reach which engine ------------------------------------------

def test_colour_matching_is_dropped_on_the_seedvr2_lane(monkeypatch):
    _settings(monkeypatch, **{'improve.colour_match': 0.8, 'improve.sharpen': 0.55,
                              'improve.grain': 0.01, 'improve.grain_saturation': 0.2})
    klein = svc._finishing_profile('klein')
    seedvr2 = svc._finishing_profile('seedvr2')

    assert klein['colour_strength'] == 0.8
    assert seedvr2['colour_strength'] == 0.0
    # The other two are engine-agnostic: neither engine does either.
    assert klein['sharpen'] == seedvr2['sharpen'] == 0.55
    assert klein['grain'] == seedvr2['grain'] == 0.01


@pytest.mark.parametrize('bad', [None, '', 'abc', {}, [], float('nan')])
def test_a_malformed_dial_degrades_to_off_not_to_a_default(monkeypatch, bad):
    """A default here would mean a broken config.json silently STARTS altering
    finished renders. Off is the only safe fallback."""
    _settings(monkeypatch, **{'improve.sharpen': bad, 'improve.grain': bad,
                              'improve.colour_match': bad})
    profile = svc._finishing_profile('klein')
    assert profile['colour_strength'] == 0.0
    assert profile['sharpen'] == 0.0
    # NaN survives min/max, so it is the one that would slip through an ordinary
    # clamp and reach the grain generator as a silent all-black image.
    assert not np.isnan(profile['grain']) and profile['grain'] == 0.0


def test_dials_are_clamped(monkeypatch):
    _settings(monkeypatch, **{'improve.colour_match': 99, 'improve.sharpen': -4,
                              'improve.grain': 99, 'improve.grain_saturation': 99})
    profile = svc._finishing_profile('klein')
    assert profile['colour_strength'] == 1.0
    assert profile['sharpen'] == 0.0
    assert profile['grain'] == 0.2
    assert profile['grain_saturation'] == 1.0


def test_the_shipped_defaults_leave_every_stage_off():
    """Ships off, so an existing install keeps the exact bytes ComfyUI wrote."""
    improve = cfg.DEFAULTS['improve']
    assert improve['colour_match'] == 0.0
    assert improve['sharpen'] == 0.0
    assert improve['grain'] == 0.0


# --- Where the hook runs -----------------------------------------------------

def test_only_an_improve_row_is_finished(monkeypatch, tmp_path):
    """A variation, a reference edit and the rescue lane all come through the very
    same callback. None of them may be touched."""
    _settings(monkeypatch, **{'improve.engine': 'klein', 'improve.sharpen': 0.55})
    for kind in (None, 'variation', svc.KLEIN_SMALL_IMAGE):
        path = tmp_path / f'{kind}.png'
        _write(path)
        before = path.read_bytes()
        svc._finish_improved_image(_Img(kind), str(path))
        assert path.read_bytes() == before, f'{kind} must not be finished'


def test_an_improve_row_is_finished(monkeypatch, tmp_path):
    _settings(monkeypatch, **{'improve.engine': 'klein', 'improve.sharpen': 0.55})
    path = tmp_path / 'improved.png'
    _write(path)
    before = path.read_bytes()

    svc._finish_improved_image(_Img(svc.KLEIN_IMAGE_IMPROVE), str(path))

    assert path.read_bytes() != before


def test_every_stage_off_does_not_even_re_encode(monkeypatch, tmp_path):
    _settings(monkeypatch, **{'improve.engine': 'klein'})
    path = tmp_path / 'improved.png'
    _write(path)
    before = path.read_bytes()

    svc._finish_improved_image(_Img(svc.KLEIN_IMAGE_IMPROVE), str(path))

    assert path.read_bytes() == before


def test_the_parent_image_is_used_as_the_colour_reference(monkeypatch, tmp_path):
    """The result must keep the grade of the image it was made FROM — so the
    reference is the parent row's file, and passing the wrong one (or none) is
    the difference between correcting drift and inventing a grade."""
    _settings(monkeypatch, **{'improve.engine': 'klein', 'improve.colour_match': 1.0})
    parent_path = tmp_path / 'parent.png'
    _write(parent_path, seed=1)

    class _Parent:
        dataset_id, filename = 'ds', 'parent.png'

    monkeypatch.setattr(svc.db.session, 'get', lambda model, ident: _Parent())
    monkeypatch.setattr(svc, '_dataset_dir', lambda dataset_id: str(tmp_path))

    seen = {}
    monkeypatch.setattr(svc.photo_finish if hasattr(svc, 'photo_finish') else pf,
                        'apply_to_file', lambda path, **kw: seen.update(kw) or True)
    import app.utils.photo_finish as real
    monkeypatch.setattr(real, 'apply_to_file',
                        lambda path, **kw: seen.update(kw) or True)

    svc._finish_improved_image(_Img(svc.KLEIN_IMAGE_IMPROVE, parent_image_id=3),
                               str(tmp_path / 'child.png'))

    assert seen['reference_path'] == os.path.join(str(tmp_path), 'parent.png')
    assert seen['colour_strength'] == 1.0


def test_the_grain_seed_is_the_row_id_so_a_rerun_is_comparable(monkeypatch, tmp_path):
    """Re-finishing the same image must not differ by noise nobody asked to
    change — otherwise an A/B of two settings also compares two grain fields."""
    _settings(monkeypatch, **{'improve.engine': 'klein', 'improve.grain': 0.01})
    seen = {}
    import app.utils.photo_finish as real
    monkeypatch.setattr(real, 'apply_to_file', lambda path, **kw: seen.update(kw) or True)

    svc._finish_improved_image(_Img(svc.KLEIN_IMAGE_IMPROVE, image_id=4242),
                               str(tmp_path / 'x.png'))
    assert seen['seed'] == 4242


def test_a_crash_in_the_pass_never_reaches_the_caller(monkeypatch, tmp_path):
    """The render is already linked by the time this runs. Losing it to a
    finishing-pass bug would be the expensive kind of wrong."""
    _settings(monkeypatch, **{'improve.engine': 'klein', 'improve.sharpen': 0.55})
    import app.utils.photo_finish as real

    def _boom(*_a, **_k):
        raise RuntimeError('nope')

    monkeypatch.setattr(real, 'apply_to_file', _boom)
    svc._finish_improved_image(_Img(svc.KLEIN_IMAGE_IMPROVE), str(tmp_path / 'x.png'))


# --- What the adversarial pass found -------------------------------------------

class _ImgWithMeta(_Img):
    def __init__(self, engine, **kw):
        super().__init__(svc.KLEIN_IMAGE_IMPROVE, **kw)
        self.generation_meta = json.dumps({'engine': engine})


def test_the_engine_is_read_off_the_row_not_the_default_setting(monkeypatch):
    """The engine is chosen PER PASS — the lightbox offers one button per engine,
    the bulk toolbar names its engine on the button — and `improve.engine` is
    only the default the single-tile button starts on. Deciding the colour match
    from the setting gave a SeedVR2 lot the Klein colour match on top of the
    node's own grading whenever the setting said 'klein' (the shipped default),
    and denied a Klein pass its match whenever it said 'seedvr2'."""
    _settings(monkeypatch, **{'improve.engine': 'klein', 'improve.colour_match': 0.8,
                              'improve.sharpen': 0.55})
    import app.utils.photo_finish as real
    seen = []
    monkeypatch.setattr(real, 'apply_to_file', lambda path, **kw: seen.append(kw) or True)

    svc._finish_improved_image(_ImgWithMeta('seedvr2'), 'x.png')
    assert seen[-1]['colour_strength'] == 0.0, 'a SeedVR2 pass must not be colour-matched'

    _settings(monkeypatch, **{'improve.engine': 'seedvr2', 'improve.colour_match': 0.8,
                              'improve.sharpen': 0.55})
    svc._finish_improved_image(_ImgWithMeta('klein'), 'x.png')
    assert seen[-1]['colour_strength'] == 0.8, 'a Klein pass keeps its colour match'


def test_a_row_without_a_stamp_falls_back_to_the_setting(monkeypatch):
    """Rows that predate the generation stamp: the setting is the only answer."""
    _settings(monkeypatch, **{'improve.engine': 'seedvr2', 'improve.colour_match': 0.8,
                              'improve.sharpen': 0.55})
    import app.utils.photo_finish as real
    seen = []
    monkeypatch.setattr(real, 'apply_to_file', lambda path, **kw: seen.append(kw) or True)
    svc._finish_improved_image(_Img(svc.KLEIN_IMAGE_IMPROVE), 'x.png')
    assert seen[-1]['colour_strength'] == 0.0


def test_a_missing_numpy_is_one_warning_not_a_traceback(monkeypatch, caplog):
    """numpy lives in requirements-ml. A user who turned finishing on without it
    gets one plain line naming the fix — never a traceback per image."""
    import logging
    import sys
    import app.utils as utils_pkg
    _settings(monkeypatch, **{'improve.engine': 'klein', 'improve.sharpen': 0.55})
    # `from ..utils import photo_finish` first looks for the attribute on the
    # package (already set once the module was imported by an earlier test), then
    # imports it — a None entry in sys.modules is how Python says "import of X
    # halted", i.e. exactly what a missing numpy produces at module load.
    monkeypatch.delattr(utils_pkg, 'photo_finish', raising=False)
    monkeypatch.setitem(sys.modules, 'app.utils.photo_finish', None)
    with caplog.at_level(logging.WARNING):
        svc._finish_improved_image(_Img(svc.KLEIN_IMAGE_IMPROVE), 'x.png')
    assert any('ML requirements' in r.getMessage() for r in caplog.records)
    assert not any(r.exc_info for r in caplog.records)
