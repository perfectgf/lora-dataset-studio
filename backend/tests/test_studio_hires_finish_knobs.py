"""The per-run hi-res fix and finishing knobs of the Studio panel: the wire, the
row, the graph, and where the finishing pass may run.

The threading bug is the one worth pinning: a per-run value dropped at any of
the hand-offs (payload -> settings object -> sanitiser -> row -> builder ->
graph) passes every unit test and renders a cell that merely ignores the panel.
So the tests below walk the value, not the function.

Three contracts that are NOT obvious from reading:

* **None means "the setting", 1.0 means "off here".** The panel has a third
  state the other knobs do not — untouched — and it must defer to Settings
  rather than send a 1.0 that would switch a Settings default off. Conversely an
  explicit 1.0 must beat a Settings 1.5, or "off for this run" is not a thing.
* **NaN is not a number.** `nan <= 1.0` is False and `min(MAX, nan)` is MAX: an
  unguarded clamp turns a corrupt value into the maximum, not into off.
* **Finishing knows which row it is on.** An ordinary cell finishes from its own
  columns and never colour-matches (text-to-image has no "before"); a ◉ Canvas
  ✨ improve finishes exactly like the dataset ✨ improve does, parent as the
  colour reference; a 📷 camera angle is left alone.
"""
import math

import pytest

from app import config as cfg


# --- the wire ----------------------------------------------------------------

def test_the_four_wire_names_survive_the_settings_object():
    from app.services.lora_test_studio import StudioGenSettings
    s = StudioGenSettings.from_payload({'hires_scale': 1.5, 'hires_denoise': 0.4,
                                        'finish_sharpen': 0.55, 'finish_grain': 0.01})
    assert (s.hires_scale, s.hires_denoise, s.finish_sharpen, s.finish_grain) == (
        1.5, 0.4, 0.55, 0.01)


def test_an_absent_field_reads_as_none_not_as_off():
    from app.services.lora_test_studio import StudioGenSettings
    s = StudioGenSettings.from_payload({})
    assert s.hires_scale is None and s.hires_denoise is None
    assert s.finish_sharpen is None and s.finish_grain is None


# --- the sanitiser -----------------------------------------------------------

def _knobs(**kw):
    from app.services.lora_test_studio import _sanitize_gen_knobs
    return _sanitize_gen_knobs('krea', **kw)


def test_hires_has_three_states_and_none_is_the_important_one():
    assert _knobs()['hires_scale'] is None                       # untouched -> setting
    assert _knobs(hires_scale=1.0)['hires_scale'] == 1.0         # off, explicitly
    assert _knobs(hires_scale=0.5)['hires_scale'] == 1.0         # shrinking is off too
    assert _knobs(hires_scale=1.5)['hires_scale'] == 1.5
    assert _knobs(hires_scale=99)['hires_scale'] == 2.0          # the injector's ceiling


@pytest.mark.parametrize('bad', ['abc', '', {}, [], float('nan'), float('inf')])
def test_a_bad_hires_value_is_untouched_never_off_and_never_maximum(bad):
    """Two silent wrong answers are possible here and both are excluded: a 1.0
    (would switch a Settings default off) and the ceiling (`min(MAX, nan)`)."""
    assert _knobs(hires_scale=bad)['hires_scale'] is None
    assert _knobs(hires_denoise=bad)['hires_denoise'] is None


def test_hires_denoise_is_clamped():
    assert _knobs(hires_denoise=0.4)['hires_denoise'] == 0.4
    assert _knobs(hires_denoise=7)['hires_denoise'] == 1.0
    assert _knobs(hires_denoise=-1)['hires_denoise'] == 0.05


def test_finishing_off_and_unset_share_one_representation():
    """Off is NULL on the row, so 0, negative, None and garbage all come back as
    None — one state, one spelling."""
    for off in (None, 0, 0.0, -1, 'abc', float('nan')):
        k = _knobs(finish_sharpen=off, finish_grain=off)
        assert k['finish_sharpen'] is None and k['finish_grain'] is None
    k = _knobs(finish_sharpen=0.55, finish_grain=0.01)
    assert (k['finish_sharpen'], k['finish_grain']) == (0.55, 0.01)
    k = _knobs(finish_sharpen=99, finish_grain=99)
    assert (k['finish_sharpen'], k['finish_grain']) == (3.0, 0.2)


def test_the_knobs_are_krea_only():
    from app.services.lora_test_studio import _sanitize_gen_knobs
    for fam in ('zimage', 'sdxl', 'flux2klein'):
        k = _sanitize_gen_knobs(fam, hires_scale=1.5, hires_denoise=0.4,
                                finish_sharpen=0.55, finish_grain=0.01)
        assert all(k[key] is None for key in
                   ('hires_scale', 'hires_denoise', 'finish_sharpen', 'finish_grain')), fam


# --- run value vs Settings default -------------------------------------------

def _settings(monkeypatch, **values):
    monkeypatch.setattr(cfg, 'get',
                        lambda dotted, default=None: values.get(dotted, default))


def test_an_untouched_run_defers_to_settings(monkeypatch):
    from app.services.lora_test_studio import _krea_hires_for_cell
    _settings(monkeypatch, **{'krea_hires.scale': 1.5, 'krea_hires.denoise': 0.5,
                              'krea_hires.steps': 0})
    assert _krea_hires_for_cell(None, None) == {
        'hires_scale': 1.5, 'hires_steps': None, 'hires_denoise': 0.5}


def test_an_explicit_off_beats_a_settings_default_that_is_on(monkeypatch):
    from app.services.lora_test_studio import _krea_hires_for_cell
    _settings(monkeypatch, **{'krea_hires.scale': 1.5, 'krea_hires.denoise': 0.5})
    assert _krea_hires_for_cell(1.0, 0.4)['hires_scale'] is None


def test_a_run_value_beats_a_settings_default_that_is_off(monkeypatch):
    from app.services.lora_test_studio import _krea_hires_for_cell
    _settings(monkeypatch, **{'krea_hires.scale': 1.0, 'krea_hires.denoise': 0.5})
    got = _krea_hires_for_cell(1.75, 0.35)
    assert (got['hires_scale'], got['hires_denoise']) == (1.75, 0.35)


def test_the_run_rewrite_applies_to_the_settings_scale_when_only_it_was_set(monkeypatch):
    """The deferred-but-on shape the panel sends: no scale, a rewrite."""
    from app.services.lora_test_studio import _krea_hires_for_cell
    _settings(monkeypatch, **{'krea_hires.scale': 1.5, 'krea_hires.denoise': 0.5})
    got = _krea_hires_for_cell(None, 0.3)
    assert (got['hires_scale'], got['hires_denoise']) == (1.5, 0.3)


def test_the_panel_defaults_are_numbers_including_the_off(monkeypatch):
    """The panel prints "Settings default (off)" from a 1.0, not from a None."""
    from app.services.lora_test_studio import krea_hires_defaults
    _settings(monkeypatch, **{'krea_hires.scale': 1.0, 'krea_hires.denoise': 0.5,
                              'krea_hires.steps': 0})
    assert krea_hires_defaults() == {'scale': 1.0, 'denoise': 0.5, 'steps': 0}
    _settings(monkeypatch, **{'krea_hires.scale': float('nan'), 'krea_hires.denoise': 'x'})
    d = krea_hires_defaults()
    assert d['scale'] == 1.0 and d['denoise'] == 0.5 and not math.isnan(d['scale'])


# --- the graph ---------------------------------------------------------------

def _krea_builder(monkeypatch):
    from app.services import lora_test_studio as lts
    monkeypatch.setattr(lts, 'get_krea_loras', lambda: [{'filename': 'krea/t.safetensors'}])
    monkeypatch.setattr(lts, 'get_krea_models', lambda: [])
    monkeypatch.setattr(lts, 'krea_default_base', lambda: None)
    return lts


def _classes(wf):
    return [n.get('class_type') for n in wf.values() if isinstance(n, dict)]


def test_a_run_value_reaches_the_built_graph_over_a_settings_off(app, monkeypatch):
    lts = _krea_builder(monkeypatch)
    _settings(monkeypatch, **{'krea_hires.scale': 1.0})
    with app.app_context():
        wf = lts._build_cell_workflow('local', 'krea/t.safetensors', 1.0, 'p', 1, None,
                                      set(), train_type='krea',
                                      hires_scale=1.5, hires_denoise=0.4)
    assert 'LatentUpscaleBy' in _classes(wf)
    assert wf['krea_hires_upscale']['inputs']['scale_by'] == 1.5
    assert wf['krea_hires_sampler']['inputs']['denoise'] == 0.4


def test_a_run_off_reaches_the_built_graph_over_a_settings_on(app, monkeypatch):
    lts = _krea_builder(monkeypatch)
    _settings(monkeypatch, **{'krea_hires.scale': 1.5, 'krea_hires.denoise': 0.5})
    with app.app_context():
        wf = lts._build_cell_workflow('local', 'krea/t.safetensors', 1.0, 'p', 1, None,
                                      set(), train_type='krea', hires_scale=1.0)
    assert 'LatentUpscaleBy' not in _classes(wf)
    assert _classes(wf).count('KSampler') == 1


def test_an_untouched_run_renders_what_settings_says(app, monkeypatch):
    lts = _krea_builder(monkeypatch)
    _settings(monkeypatch, **{'krea_hires.scale': 1.5, 'krea_hires.denoise': 0.5})
    with app.app_context():
        wf = lts._build_cell_workflow('local', 'krea/t.safetensors', 1.0, 'p', 1, None,
                                      set(), train_type='krea')
    assert wf['krea_hires_upscale']['inputs']['scale_by'] == 1.5


# --- the row -----------------------------------------------------------------

def test_the_new_columns_are_in_the_additive_migration():
    """Existing databases get them on boot, NULL — which the builder reads as
    "the setting" and the finisher as "off": an old cell resumes unchanged."""
    from app import _SCHEMA_ADDITIONS
    for col in ('hires_scale', 'hires_denoise', 'finish_sharpen', 'finish_grain'):
        assert ('lora_test_image', col, 'REAL') in _SCHEMA_ADDITIONS, col


def test_a_run_persists_its_knobs_on_every_cell(app, monkeypatch):
    """create_run, stubbed at the enqueue and the build: the thing under test is
    the column write, and it is the hand-off a threading bug drops."""
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.models import LoraTestImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Krea', 'kt')
        ck = 'krea\\lora_kt_000001000.safetensors'
        monkeypatch.setattr(lts, 'list_test_checkpoints', lambda _ds, _f=None: [{'filename': ck}])
        monkeypatch.setattr(lts, 'get_krea_models', lambda: [])
        monkeypatch.setattr(lts, 'permanent_lora_candidates', lambda _f: [])
        monkeypatch.setattr(lts, 'gpu_busy_reason', lambda: None)
        monkeypatch.setattr(lts, '_active_run_count', lambda *a: 0)
        monkeypatch.setattr(lts, '_preflight_checkpoint_arch', lambda *a, **k: None)
        monkeypatch.setattr(lts, '_preflight_run', lambda *a, **k: None)
        monkeypatch.setattr(lts, '_target_node_classes', lambda: None)
        built = []
        monkeypatch.setattr(lts, '_build_cell_workflow',
                            lambda *a, **k: built.append(k) or {'1': {}})
        monkeypatch.setattr(lts, '_enqueue_cell', lambda *a, job_id=None, **k: job_id)

        out = lts.create_run(
            LOCAL_USER, ds.id, [ck], [1.0],
            lts.StudioGenSettings(prompt='p', count=1, hires_scale=1.5, hires_denoise=0.4,
                                  finish_sharpen=0.55, finish_grain=0.01),
            family='krea')

        rows = LoraTestImage.query.filter_by(run_id=out['run_id']).all()
        assert rows
        for r in rows:
            assert (r.hires_scale, r.hires_denoise) == (1.5, 0.4)
            assert (r.finish_sharpen, r.finish_grain) == (0.55, 0.01)
        # ...and the builder was handed the same values the row keeps.
        assert built and all(b['hires_scale'] == 1.5 and b['hires_denoise'] == 0.4
                             for b in built)


# --- where finishing runs ----------------------------------------------------

class _Row:
    def __init__(self, kind=None, **attrs):
        self.derivation_kind = kind
        self.id = 5
        self.parent_image_id = None
        self.improve_profile = None
        self.finish_sharpen = None
        self.finish_grain = None
        self.__dict__.update(attrs)


def _capture(monkeypatch):
    import app.utils.photo_finish as pf
    seen = []
    monkeypatch.setattr(pf, 'apply_to_file', lambda path, **kw: seen.append(kw) or True)
    return seen


def test_an_ordinary_cell_finishes_from_its_own_columns_and_never_colour_matches(monkeypatch):
    from app.services import lora_test_studio as lts
    seen = _capture(monkeypatch)
    lts._finish_test_image(_Row(finish_sharpen=0.55, finish_grain=0.01), 'x.png')
    assert len(seen) == 1
    assert (seen[0]['sharpen'], seen[0]['grain']) == (0.55, 0.01)
    assert 'reference_path' not in seen[0] and 'colour_strength' not in seen[0]


def test_an_ordinary_cell_with_nothing_set_is_left_alone(monkeypatch):
    from app.services import lora_test_studio as lts
    seen = _capture(monkeypatch)
    lts._finish_test_image(_Row(), 'x.png')
    assert seen == []


def test_a_canvas_improve_finishes_like_the_dataset_improve(monkeypatch, tmp_path):
    """Same button, same result — improve.* from Settings, parent as reference,
    engine read off the row (improve_profile present = Klein)."""
    from app.services import lora_test_studio as lts, face_dataset_service as fds
    _settings(monkeypatch, **{'improve.colour_match': 0.8, 'improve.sharpen': 0.55,
                              'improve.grain': 0.01, 'improve.grain_saturation': 0.2})
    parent_file = tmp_path / 'parent.png'
    parent_file.write_bytes(b'x')

    class _Parent:
        dataset_id, filename = 'ds', 'parent.png'

    monkeypatch.setattr(lts.db.session, 'get', lambda model, ident: _Parent())
    monkeypatch.setattr(fds, '_dataset_dir', lambda dataset_id: str(tmp_path))
    seen = _capture(monkeypatch)

    lts._finish_test_image(_Row(lts.CANVAS_IMAGE_IMPROVE, parent_image_id=3,
                                improve_profile='{"engine": "klein"}'), 'child.png')

    assert len(seen) == 1
    assert seen[0]['colour_strength'] == 0.8
    assert seen[0]['reference_path'] == str(parent_file)


def test_a_canvas_seedvr2_improve_gets_no_colour_match(monkeypatch):
    """improve_profile is NULL for SeedVR2 — and SeedVR2 already grades back
    onto its source inside the node."""
    from app.services import lora_test_studio as lts
    _settings(monkeypatch, **{'improve.colour_match': 0.8, 'improve.sharpen': 0.55})
    seen = _capture(monkeypatch)
    lts._finish_test_image(_Row(lts.CANVAS_IMAGE_IMPROVE, parent_image_id=3), 'child.png')
    assert len(seen) == 1
    assert seen[0]['colour_strength'] == 0.0 and seen[0]['sharpen'] == 0.55


def test_a_camera_angle_row_is_left_alone(monkeypatch):
    from app.services import lora_test_studio as lts
    _settings(monkeypatch, **{'improve.sharpen': 0.55})
    seen = _capture(monkeypatch)
    lts._finish_test_image(_Row(lts.CAMERA_ANGLE, finish_sharpen=0.55), 'x.png')
    assert seen == []


def test_a_crash_in_finishing_never_reaches_the_link(monkeypatch):
    from app.services import lora_test_studio as lts
    import app.utils.photo_finish as pf

    def _boom(*_a, **_k):
        raise RuntimeError('nope')

    monkeypatch.setattr(pf, 'apply_to_file', _boom)
    lts._finish_test_image(_Row(finish_sharpen=0.55), 'x.png')
