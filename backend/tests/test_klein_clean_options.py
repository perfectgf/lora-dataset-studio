"""The 🧽 Klein clean's three user-facing dials: the PROMPT that is actually sent, the
megapixel CAP the frame travels at, and the WRITE-BACK size of the file.

Before 2026-08-31 all three were constants in the source. "remove watermark" was
therefore invisible from the app — a user whose mark survived had nothing to turn — and
2 MP was a decision nobody could revisit on a 24 GB card. They are config now
(`watermark_clean.*`), read on both surfaces, and this file pins what that has to mean.

The trap these tests exist to hold shut: inside `inpaint_watermark_klein` the `prompt`
argument is the LANE SWITCH (given → the ✦ crop repair, empty → this clean). The
editable clean instruction travels as its own `klein_prompt` and must never be routed
into it, or every clean silently becomes a crop repair that preserves the watermark
everywhere outside the boxes.
"""
import logging

import pytest
from PIL import Image, ImageDraw


def _photo(w, h):
    """Four different quadrants, so "the whole frame was sent" is provable rather than
    assumed: any crop of it loses at least one corner colour."""
    img = Image.new('RGB', (w, h), (120, 120, 120))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w // 2 - 1, h // 2 - 1], fill=(200, 40, 40))
    d.rectangle([w // 2, 0, w - 1, h // 2 - 1], fill=(40, 200, 40))
    d.rectangle([0, h // 2, w // 2 - 1, h - 1], fill=(40, 40, 200))
    d.rectangle([w // 2, h // 2, w - 1, h - 1], fill=(220, 220, 40))
    return img


@pytest.fixture
def klein_clean(monkeypatch):
    """The clean lane with its two outside edges captured: the erase step is a
    pass-through that records nothing, and the ComfyUI round-trip is a fake that records
    exactly what it was handed. Everything else — sizing, prompt resolution, write-back
    — is the real code."""
    from app.services import watermark_klein as wk
    monkeypatch.setattr(wk, 'is_available', lambda: True)
    monkeypatch.setattr(wk, '_prefill_region',
                        lambda frame, boxes, device='cpu': (frame, None))
    sent = {}

    def _fake_klein(user_id, frame, *, seed, timeout=None, **kw):
        sent['size'] = frame.size
        sent['kw'] = kw
        # A real render comes back at the size it was sent at.
        return frame.copy(), None

    monkeypatch.setattr(wk, '_run_klein_job', _fake_klein)
    return wk, sent


def _stored(monkeypatch, **values):
    """Pin `watermark_clean.*` config values. Anything not named reads as absent, which
    is the shape of every config written before this section existed."""
    from app import config as cfg
    monkeypatch.setattr(cfg, 'get', lambda key, default=None: values.get(key, default))


# --- The prompt ---------------------------------------------------------------

def test_the_clean_sends_the_stored_prompt(app, klein_clean, monkeypatch, tmp_path):
    """What the user typed is what reaches ComfyUI — the whole point of making it
    visible. `_run_klein_job(prompt=...)` is the seam, so it is asserted there."""
    wk, sent = klein_clean
    _stored(monkeypatch, **{'watermark_clean.klein_prompt': '  erase every logo  '})
    img = tmp_path / 'wm.webp'
    _photo(320, 240).save(img, 'WEBP', lossless=True)

    with app.app_context():
        ok, err = wk.inpaint_watermark_klein('local', str(img), [[0.4, 0.4, 0.5, 0.5]], seed=3)

    assert ok and err is None
    # trimmed, not reformatted: leading/trailing blanks in a text box are typing, not intent
    assert sent['kw']['prompt'] == 'erase every logo'


def test_an_empty_prompt_falls_back_to_the_shipped_three_words(app, klein_clean,
                                                               monkeypatch, tmp_path):
    """A blank field means "the default", not "send nothing". config.json is
    hand-editable and the text box can be emptied with one keystroke; either must leave
    a user with a working clean rather than a naked render."""
    wk, sent = klein_clean
    for stored in ('', '   ', None):
        _stored(monkeypatch, **{'watermark_clean.klein_prompt': stored})
        img = tmp_path / 'wm.webp'
        _photo(320, 240).save(img, 'WEBP', lossless=True)
        with app.app_context():
            ok, err = wk.inpaint_watermark_klein('local', str(img), [[0.4, 0.4, 0.5, 0.5]])
        assert ok and err is None
        assert sent['kw']['prompt'] == wk.KLEIN_CLEAN_PROMPT == 'remove watermark'


def test_the_editable_prompt_never_reaches_the_repair_lane(app, klein_clean,
                                                           monkeypatch, tmp_path):
    """THE regression this whole design is shaped around.

    `prompt` switches lanes; `klein_prompt` is the clean's text. Wire the second into the
    first and a 🧽 clean becomes a ✦ crop repair: it would report success, preserve every
    pixel outside the detected boxes, and leave a tiled mark exactly where it was — with
    the detector agreeing, because the boxes it knew about really were repainted."""
    wk, sent = klein_clean
    monkeypatch.setattr(wk, '_repair_boxes_crop_and_stitch', lambda *a, **k: (_ for _ in ()).throw(
        AssertionError('the clean took the ✦ repair lane — klein_prompt leaked into `prompt`')))
    _stored(monkeypatch, **{'watermark_clean.klein_prompt': 'wipe it'})
    img = tmp_path / 'wm.webp'
    _photo(320, 240).save(img, 'WEBP', lossless=True)

    with app.app_context():
        # both routes into the clean: the stored value, and an explicit per-run override
        for override in (None, 'wipe it harder'):
            ok, err = wk.inpaint_watermark_klein('local', str(img), [[0.1, 0.1, 0.3, 0.3]],
                                                 klein_prompt=override)
            assert ok and err is None
    assert sent['kw']['prompt'] == 'wipe it harder'


# --- The processing size ------------------------------------------------------

def test_clean_max_mp_clamps_and_falls_back(monkeypatch):
    """Pure resolver. A hand-edited config must never be able to send a 40 MP frame to a
    GPU that also holds ComfyUI, nor a 0.01 MP one that comes back as mush."""
    from app.services import watermark_klein as wk
    _stored(monkeypatch)                       # nothing stored at all
    assert wk.clean_max_mp() == wk.KLEIN_MASK_MAX_MP == 2.0
    assert wk.clean_max_mp(12) == wk.KLEIN_CLEAN_MAX_MP_MAX == 4.0
    assert wk.clean_max_mp(0.01) == wk.KLEIN_CLEAN_MAX_MP_MIN == 0.5
    assert wk.clean_max_mp(3) == 3.0
    assert wk.clean_max_mp('1.5') == 1.5       # config.json hand-edited as a string
    # Garbage is "the default", never an exception: nobody debugs a clean that refuses
    # to start because a config value has a typo in it.
    for junk in (None, 'abc', [], {}, float('nan'), float('inf')):
        assert wk.clean_max_mp(junk) == 2.0
    # `float(True)` is 1.0 — a JSON `true` silently becoming a 1 MP cap is unfindable.
    assert wk.clean_max_mp(True) == 2.0
    assert wk.clean_max_mp(False) == 2.0
    # stored value, and the override that beats it
    _stored(monkeypatch, **{'watermark_clean.klein_max_mp': 3})
    assert wk.clean_max_mp() == 3.0
    assert wk.clean_max_mp(1) == 1.0


def test_the_frame_travels_at_the_stored_cap(app, klein_clean, monkeypatch, tmp_path):
    """The cap is what actually sizes the frame handed to Klein — asserted on the image
    that reaches the round-trip, not on the resolver."""
    wk, sent = klein_clean
    img = tmp_path / 'wm.webp'
    _photo(1600, 1200).save(img, 'WEBP', lossless=True)   # 1.92 MP

    for stored, expected_cap in ((1.0, 1.0), (4.0, 4.0), (99, 4.0)):
        _stored(monkeypatch, **{'watermark_clean.klein_max_mp': stored})
        with app.app_context():
            ok, err = wk.inpaint_watermark_klein('local', str(img), [[0.4, 0.4, 0.5, 0.5]])
        assert ok and err is None
        assert sent['size'] == wk._mask_frame_size(1600, 1200, max_mp=expected_cap)
    # 1 MP scales a 1.92 MP photo down; 4 MP does NOT magnify it — a bigger cap must
    # never invent detail the file does not have.
    _stored(monkeypatch, **{'watermark_clean.klein_max_mp': 1.0})
    with app.app_context():
        wk.inpaint_watermark_klein('local', str(img), [[0.4, 0.4, 0.5, 0.5]])
    small = sent['size']
    _stored(monkeypatch, **{'watermark_clean.klein_max_mp': 4.0})
    with app.app_context():
        wk.inpaint_watermark_klein('local', str(img), [[0.4, 0.4, 0.5, 0.5]])
    assert small[0] < sent['size'][0] <= 1600 and sent['size'][1] <= 1200


def test_the_masked_repair_lane_keeps_its_own_fixed_cap(monkeypatch):
    """`_mask_frame_size` is SHARED with the masked repair, which composites its render
    back and therefore gains nothing from a bigger frame outside the mask. The clean's
    dial must not reach it: its default argument stays the constant."""
    from app.services import watermark_klein as wk
    _stored(monkeypatch, **{'watermark_clean.klein_max_mp': 4.0})
    assert wk._mask_frame_size(6000, 4000) == wk._mask_frame_size(
        6000, 4000, max_mp=wk.KLEIN_MASK_MAX_MP)


# --- The write-back size ------------------------------------------------------

def test_write_back_original_keeps_the_files_own_dimensions(app, klein_clean,
                                                            monkeypatch, tmp_path):
    """What shipped, and the default: a clean never changes the shape of a dataset
    image, whatever size the render came back at."""
    wk, sent = klein_clean
    _stored(monkeypatch, **{'watermark_clean.klein_max_mp': 1.0,
                            'watermark_clean.klein_output': 'original'})
    img = tmp_path / 'wm.webp'
    _photo(1600, 1200).save(img, 'WEBP', lossless=True)

    with app.app_context():
        ok, err = wk.inpaint_watermark_klein('local', str(img), [[0.4, 0.4, 0.5, 0.5]])

    assert ok and err is None
    assert sent['size'] != (1600, 1200), 'the 1 MP cap should have scaled the frame down'
    with Image.open(img) as out:
        assert out.size == (1600, 1200)


def test_write_back_render_writes_the_render_size(app, klein_clean, monkeypatch,
                                                  tmp_path):
    """The opt-in: no second resample, so the file ENDS UP SMALLER. That is the whole
    trade and every surface that offers it says so — this test is what makes sure the
    behaviour matches the warning."""
    wk, sent = klein_clean
    _stored(monkeypatch, **{'watermark_clean.klein_max_mp': 1.0,
                            'watermark_clean.klein_output': 'render'})
    img = tmp_path / 'wm.webp'
    _photo(1600, 1200).save(img, 'WEBP', lossless=True)

    with app.app_context():
        ok, err = wk.inpaint_watermark_klein('local', str(img), [[0.4, 0.4, 0.5, 0.5]])

    assert ok and err is None
    with Image.open(img) as out:
        assert out.size == sent['size'] == wk._mask_frame_size(1600, 1200, max_mp=1.0)
        assert out.size != (1600, 1200)


def test_write_back_render_changes_nothing_below_the_cap(app, klein_clean, monkeypatch,
                                                         tmp_path):
    """`_mask_frame_size` never magnifies, so on a photo already under the cap the two
    modes are the same file. Worth pinning: it is why the option is honest to offer as a
    default-off choice rather than a scary one."""
    wk, _sent = klein_clean
    img = tmp_path / 'wm.webp'
    _photo(640, 480).save(img, 'WEBP', lossless=True)     # 0.3 MP, well under any cap
    for mode in ('original', 'render'):
        _stored(monkeypatch, **{'watermark_clean.klein_output': mode})
        with app.app_context():
            ok, err = wk.inpaint_watermark_klein('local', str(img), [[0.4, 0.4, 0.5, 0.5]])
        assert ok and err is None
        with Image.open(img) as out:
            assert out.size == (640, 480)


def test_clean_output_mode_rejects_anything_it_does_not_know(monkeypatch):
    """An unknown mode resolves to 'original' — the mode that cannot surprise anyone by
    resizing their files."""
    from app.services import watermark_klein as wk
    _stored(monkeypatch)
    assert wk.clean_output_mode() == wk.KLEIN_CLEAN_OUTPUT_DEFAULT == 'original'
    assert wk.clean_output_mode('  RENDER ') == 'render'
    for junk in ('', 'native', 'full', None, 3):
        assert wk.clean_output_mode(junk) == 'original'
    _stored(monkeypatch, **{'watermark_clean.klein_output': 'render'})
    assert wk.clean_output_mode() == 'render'
    assert wk.clean_output_mode('original') == 'original'


# --- What the run leaves behind -----------------------------------------------

def test_the_run_log_names_the_prompt_that_actually_ran(app, klein_clean, monkeypatch,
                                                        tmp_path, caplog):
    """The prompt is editable, so the source no longer answers "what cleaned this
    batch?". The run has to. Logged BEFORE the render, so a job that fails or times out
    still says what it was asked to do."""
    wk, _sent = klein_clean
    _stored(monkeypatch, **{'watermark_clean.klein_prompt': 'delete the signature',
                            'watermark_clean.klein_max_mp': 1.0,
                            'watermark_clean.klein_output': 'render'})
    img = tmp_path / 'wm.webp'
    _photo(1600, 1200).save(img, 'WEBP', lossless=True)

    with caplog.at_level(logging.INFO, logger='app.services.watermark_klein'):
        with app.app_context():
            ok, err = wk.inpaint_watermark_klein('local', str(img), [[0.4, 0.4, 0.5, 0.5]])

    assert ok and err is None
    line = '\n'.join(r.getMessage() for r in caplog.records)
    assert 'delete the signature' in line
    assert 'render' in line                    # the write-back mode
    assert '1.00 MP' in line                   # the cap that sized the frame
    # No path: this line ends up in pasted diagnostics (CLAUDE.md, privacy).
    assert str(tmp_path) not in line and 'wm.webp' not in line


def test_a_failed_render_still_logged_what_it_was_asked_to_do(app, monkeypatch,
                                                              tmp_path, caplog):
    """The failure case is the one where the question gets asked, so the log line cannot
    live after the round-trip."""
    from app.services import watermark_klein as wk
    monkeypatch.setattr(wk, 'is_available', lambda: True)
    monkeypatch.setattr(wk, '_prefill_region', lambda frame, boxes, device='cpu': (frame, None))
    monkeypatch.setattr(wk, '_run_klein_job',
                        lambda *a, **k: (None, {'kind': 'failed', 'detail': 'timeout'}))
    _stored(monkeypatch, **{'watermark_clean.klein_prompt': 'scrub the mark'})
    img = tmp_path / 'wm.webp'
    _photo(320, 240).save(img, 'WEBP', lossless=True)

    with caplog.at_level(logging.INFO, logger='app.services.watermark_klein'):
        with app.app_context():
            ok, err = wk.inpaint_watermark_klein('local', str(img), [[0.4, 0.4, 0.5, 0.5]])

    assert ok is False and err == {'kind': 'failed', 'detail': 'timeout'}
    assert 'scrub the mark' in '\n'.join(r.getMessage() for r in caplog.records)


# --- What the front is told ---------------------------------------------------

def test_capabilities_publish_the_RESOLVED_dials_not_the_raw_config(monkeypatch):
    """Both surfaces quote these values back to the user before they launch a run, so
    they must be what the pass will really do. A hand-edited `klein_max_mp: 12` that the
    service clamps to 4 must not be shown as 12 — the same reason
    `watermark_detect_threshold` is published resolved."""
    from app import capabilities
    _stored(monkeypatch, **{'watermark_clean.klein_prompt': '  take the logo out ',
                            'watermark_clean.klein_max_mp': 12,
                            'watermark_clean.klein_output': 'RENDER'})
    assert capabilities._watermark_clean_options() == {
        'prompt': 'take the logo out', 'max_mp': 4.0, 'output': 'render'}
    _stored(monkeypatch)
    assert capabilities._watermark_clean_options() == {
        'prompt': 'remove watermark', 'max_mp': 2.0, 'output': 'original'}


def test_the_config_section_ships_the_documented_defaults():
    """The stored defaults and the code's fallbacks are two different places, and a
    divergence would show as a Settings screen disagreeing with the pass it describes."""
    from app import config as cfg
    from app.services import watermark_klein as wk
    section = cfg.DEFAULTS['watermark_clean']
    assert section['klein_prompt'] == wk.KLEIN_CLEAN_PROMPT
    assert section['klein_max_mp'] == wk.KLEIN_MASK_MAX_MP
    assert section['klein_output'] == wk.KLEIN_CLEAN_OUTPUT_DEFAULT
    assert set(section) == {'klein_prompt', 'klein_max_mp', 'klein_output'}
