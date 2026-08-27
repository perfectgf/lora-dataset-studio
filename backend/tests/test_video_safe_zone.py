"""🔳 The safe zone — bands, burned-in text, and what is left of the frame.

What is under test is NOT "does RapidOCR read letters" (it is someone else's
checkpoint and it cannot be imported in this interpreter) but the decisions that
make this safe to run over somebody's bank:

  * the BAND RULE is uniformity, not darkness, so a grey or white pillarbox is
    found — and a frame that is uniform all over ABSTAINS rather than claiming to
    be all container;
  * the VOTE is a per-side MIN across the shot's frames, which is the only
    aggregation whose answer is true of every frame — the property a crop needs;
  * a text zone is STRUCTURAL only when it appears in two frames, which is what
    separates a subtitle from a shop sign, and boxes in the SAME frame are never
    merged;
  * `text_coverage` is a UNION, not a sum: two OCR boxes over one line of
    subtitle must not count twice;
  * the SAFE ZONE is an edge crop, cheapest cut first — a bottom subtitle costs
    its own height, and text in the middle of the frame honestly collapses it;
  * every degraded case is a STATE and never a zero: no frame decoded is
    'unreadable', no OCR engine is 'bars_only' with the text keys ABSENT, and an
    OCR run that did not reach a shot leaves it in the queue;
  * the verdicts are derived at READ time, so moving a cut re-sorts the bank with
    nothing rescanned.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.services import video_metrics
from app.services import video_safe_zone as sz
from app.services import video_safe_zone_geometry as geo


# --- the band rule ------------------------------------------------------------------

def _grid(width, height, fill=120):
    return [[fill] * width for _ in range(height)]


def _noise(width, height, base=90):
    """An interior no uniformity test can mistake for a band: every row and every
    column carries a spread far wider than BAND_TOLERANCE."""
    return [[base + ((x * 7 + y * 13) % 120) for x in range(width)]
            for y in range(height)]


def _letterboxed(width, height, band, level=0):
    rows = _noise(width, height)
    for y in range(band):
        rows[y] = [level] * width
        rows[height - 1 - y] = [level] * width
    return rows


def _pillarboxed(width, height, band, level=0):
    rows = _noise(width, height)
    for row in rows:
        for x in range(band):
            row[x] = level
            row[width - 1 - x] = level
    return rows


def test_black_letterbox_bands_are_measured_per_side():
    bands = geo.bands_of_grid(_letterboxed(100, 100, 12, level=0))
    assert bands == {'top': 0.12, 'bottom': 0.12, 'left': 0.0, 'right': 0.0}


def test_a_uniform_band_that_is_not_black_is_still_a_band():
    """The image lane tests DARKNESS (`max(...) <= _BARS_DARK`), which is right
    for stills off a chat app and blind to the way rushes are padded: an editor's
    white surround, a grey mat, a scan of a 4:3 broadcast. Uniformity covers all
    of them with one rule — measured here on white AND mid-grey."""
    for level in (255, 128):
        bands = geo.bands_of_grid(_pillarboxed(100, 100, 8, level=level))
        assert bands == {'top': 0.0, 'bottom': 0.0, 'left': 0.08, 'right': 0.08}, \
            f'a band at luma {level} was missed'


def test_a_frame_with_bands_on_both_axes_is_measured_on_both():
    rows = _letterboxed(100, 100, 10, level=0)
    for row in rows:
        for x in range(6):
            row[x] = 0
            row[99 - x] = 0
    bands = geo.bands_of_grid(rows)
    assert bands == {'top': 0.1, 'bottom': 0.1, 'left': 0.06, 'right': 0.06}


def test_a_frame_full_of_picture_has_no_bands():
    assert geo.bands_of_grid(_noise(100, 100)) == {
        'top': 0.0, 'bottom': 0.0, 'left': 0.0, 'right': 0.0}


def test_a_uniform_frame_abstains_rather_than_claiming_to_be_all_container():
    """A fade-out, a black slug, a plain title card. Zeros would claim it is full
    of picture and the caps would let one fading frame shrink a whole shot's safe
    zone — neither is true, so the frame does not vote at all."""
    assert geo.bands_of_grid(_grid(100, 100, fill=0)) is None
    assert geo.bands_of_grid(_grid(100, 100, fill=255)) is None


def test_a_band_can_never_claim_more_than_the_cap():
    """A near-uniform frame with one row of detail in the middle: the scan walks
    until MAX_BAND and stops. Without the cap a single dark frame reports a 100 %
    letterbox, which is a measurement about a fade rather than a container."""
    rows = _grid(100, 100, fill=0)
    rows[50] = [10 + (x * 5) % 200 for x in range(100)]
    bands = geo.bands_of_grid(rows)
    assert bands is not None
    assert bands['top'] == pytest.approx(geo.MAX_BAND, abs=0.01)
    assert bands['bottom'] == pytest.approx(geo.MAX_BAND, abs=0.01)


# --- the vote -----------------------------------------------------------------------

def test_a_band_on_every_frame_is_structural():
    three = [{'top': 0.12, 'bottom': 0.12, 'left': 0.0, 'right': 0.0}] * 3
    assert geo.vote_bands(three)['top'] == 0.12


def test_a_band_on_one_frame_only_is_a_transition_and_not_a_container():
    """The failure the vote exists to prevent: a shot whose middle frame fades
    would otherwise carry a permanent phantom letterbox nobody can see."""
    voted = geo.vote_bands([
        {'top': 0.0, 'bottom': 0.0, 'left': 0.0, 'right': 0.0},
        {'top': 0.40, 'bottom': 0.40, 'left': 0.0, 'right': 0.0},
        {'top': 0.0, 'bottom': 0.0, 'left': 0.0, 'right': 0.0}])
    assert voted == {'top': 0.0, 'bottom': 0.0, 'left': 0.0, 'right': 0.0}


def test_the_vote_never_claims_more_band_than_the_least_banded_frame():
    """The property that makes a crop from this SAFE: whatever comes out is true
    of every frame, so nothing the crop keeps was ever container."""
    voted = geo.vote_bands([
        {'top': 0.20, 'bottom': 0.10, 'left': 0.0, 'right': 0.0},
        {'top': 0.12, 'bottom': 0.14, 'left': 0.0, 'right': 0.0}])
    assert voted == {'top': 0.12, 'bottom': 0.10, 'left': 0.0, 'right': 0.0}


def test_frames_that_abstained_do_not_vote_and_all_of_them_abstaining_is_none():
    kept = geo.vote_bands([None, {'top': 0.1, 'bottom': 0.1,
                                  'left': 0.0, 'right': 0.0}, None])
    assert kept['top'] == 0.1
    assert geo.vote_bands([None, None]) is None
    assert geo.vote_bands([]) is None


def test_the_banded_share_is_the_larger_axis_and_matches_the_image_lane():
    """Same arithmetic, and therefore the same MEANING, as
    image_provenance.bars_ratio — so `0.12` calibrated on stills carries over.
    Read off that module's source rather than asserted from memory."""
    bands = {'top': 0.12, 'bottom': 0.12, 'left': 0.03, 'right': 0.03}
    assert geo.bars_ratio(bands) == pytest.approx(0.24)

    src = (Path(__file__).resolve().parents[1] / 'app' / 'services'
           / 'image_provenance.py').read_text(encoding='utf-8')
    assert 'max((top + bottom) / h, (left + right) / w)' in src, \
        ('the image lane changed how it folds four sides into one number — the '
         'video cut carries the same name and must not mean something else')


def test_the_per_side_bands_survive_any_alignment_with_the_image_lane():
    """The half a future "unification" would delete, pinned.

    The two lanes share the WORD `bars_ratio` and the arithmetic that produces
    it, and a reader who notices that will reasonably reach for the merge. The
    image lane stores ONE scalar; this one must keep FOUR fractions, because
    `safe_rect` has to know WHICH edges to cut and a scalar cannot say whether
    0.24 is a letterbox or a pillarbox — two clips with the identical ratio and
    opposite geometry, here, to make the point undeniable."""
    letterbox = {'top': 0.12, 'bottom': 0.12, 'left': 0.0, 'right': 0.0}
    pillarbox = {'top': 0.0, 'bottom': 0.0, 'left': 0.12, 'right': 0.12}
    assert geo.bars_ratio(letterbox) == geo.bars_ratio(pillarbox)
    # Same scalar, opposite crops. The per-side values are the only thing that
    # tells them apart, and the safe zone is built on them.
    assert geo.safe_rect(letterbox, []) == (0.0, 0.12, 1.0, 0.88)
    assert geo.safe_rect(pillarbox, []) == (0.12, 0.0, 0.88, 1.0)
    assert geo.SIDES == ('top', 'bottom', 'left', 'right')


# --- burned-in text -----------------------------------------------------------------

SUB_A = (0.20, 0.82, 0.80, 0.90)        # a subtitle line
SUB_B = (0.24, 0.82, 0.76, 0.90)        # the next line: same place, other words
SIGN = (0.05, 0.20, 0.18, 0.28)         # a shop sign, seen once


def test_a_zone_in_two_frames_is_burned_into_the_footage():
    found = geo.structural_text([[SUB_A], [SUB_B], [SUB_A]])
    assert len(found) == 1
    assert found[0] == pytest.approx((0.20, 0.82, 0.80, 0.90))


def test_a_zone_in_one_frame_only_is_scene_content_and_is_not_flagged():
    """The whole point of paying for three frames. A shop sign, a newspaper held
    up for an instant, a licence plate that pans through — footage, not overlay,
    and cropping it away would be destroying the shot to save it."""
    assert geo.structural_text([[SIGN], [], []]) == []


def test_a_one_frame_shot_reports_its_text_rather_than_a_hard_zero():
    """Shots under MIN_SPAN_FOR_THREE_S contribute ONE frame, so the two-frame
    rule can never be met and an unclamped version would report every sub-second
    shot as carrying no burned text — a hard zero about footage it never got to
    compare. On one frame the visible mistake is the right one: a shop sign
    raises an advisory flag somebody glances at, a silent subtitle is never
    seen."""
    assert geo.structural_text([[SUB_A]]) == [SUB_A]
    # Two frames still need both — the clamp never makes the rule LOOSER than
    # the evidence allows.
    assert geo.structural_text([[SIGN], []]) == []


def test_a_drifting_subtitle_is_reported_as_the_box_that_covers_all_of_it():
    """The bounding box, never an average: a safe zone built on the mean of two
    readings leaves a sliver of letters inside itself, and a sliver of letters
    trains exactly as well as a whole one."""
    found = geo.structural_text([[(0.2, 0.80, 0.7, 0.88)],
                                 [(0.22, 0.82, 0.78, 0.90)]])
    assert found == [(0.2, 0.80, 0.78, 0.90)]


def test_two_zones_stacked_on_one_frame_never_merge_into_one_tall_box():
    """Linking happens across frames only. A two-line subtitle is two zones, and
    merging them here would report a box covering the gap between them."""
    top_line = (0.2, 0.74, 0.8, 0.80)
    bottom_line = (0.2, 0.82, 0.8, 0.88)
    found = geo.structural_text([[top_line, bottom_line],
                                 [top_line, bottom_line]])
    assert found == [top_line, bottom_line]


def test_the_covered_share_is_a_union_and_never_double_counts():
    """Two OCR boxes over one subtitle line overlap heavily. Summing them would
    flag a shot whose text takes 3 % of the frame at a cut set for 5 %."""
    one = (0.0, 0.0, 0.2, 0.2)          # 0.04
    overlapping = (0.1, 0.1, 0.3, 0.3)  # 0.04, sharing 0.01
    assert geo.union_area([one, overlapping]) == pytest.approx(0.07)
    assert geo.union_area([one, one]) == pytest.approx(0.04)
    assert geo.union_area([]) == 0.0
    # Disjoint boxes still add up, and a box wholly inside another adds nothing.
    assert geo.union_area([(0, 0, 0.1, 0.1), (0.5, 0.5, 0.6, 0.6)]) \
        == pytest.approx(0.02)
    assert geo.union_area([(0, 0, 0.4, 0.4), (0.1, 0.1, 0.2, 0.2)]) \
        == pytest.approx(0.16)


def test_the_covered_share_stays_cheap_on_a_frame_full_of_text():
    """A screen recording, a scoreboard, a scan of a printed page: RapidOCR
    returns dozens of boxes and every one of them can be structural. The first
    draft compressed coordinates into a grid, which is O(n³) — at this size that
    is hundreds of millions of point-in-box tests and one clip freezing the pass
    for minutes. Pinned as a BUDGET, not a benchmark: the failure it guards
    against is three orders of magnitude away, so this cannot flake on a busy
    machine."""
    import time
    boxes = [(0.01 * (i % 10), 0.01 * (i // 10),
              0.01 * (i % 10) + 0.35, 0.01 * (i // 10) + 0.35)
             for i in range(100)]
    start = time.perf_counter()
    area = geo.union_area(boxes)
    elapsed = time.perf_counter() - start
    assert 0.0 < area <= 1.0
    assert elapsed < 2.0, f'100 overlapping zones took {elapsed:.1f}s'


# --- the safe zone ------------------------------------------------------------------

def test_a_bottom_subtitle_costs_its_own_height_and_not_half_the_picture():
    rect = geo.safe_rect(None, [(0.2, 0.82, 0.8, 0.90)])
    assert rect == (0.0, 0.0, 1.0, 0.82)
    assert geo.safe_area(rect) == pytest.approx(0.82)


def test_the_bands_are_excluded_before_any_text_is_considered():
    bands = {'top': 0.12, 'bottom': 0.12, 'left': 0.0, 'right': 0.0}
    rect = geo.safe_rect(bands, [])
    assert rect == (0.0, 0.12, 1.0, 0.88)
    assert geo.safe_area(rect) == pytest.approx(0.76)


def test_bands_and_a_subtitle_inside_them_are_both_removed():
    """The realistic shape: a letterboxed film with the subtitle sitting just
    above the lower band."""
    bands = {'top': 0.12, 'bottom': 0.12, 'left': 0.0, 'right': 0.0}
    rect = geo.safe_rect(bands, [(0.2, 0.80, 0.8, 0.86)])
    assert rect == (0.0, 0.12, 1.0, 0.80)
    assert geo.safe_area(rect) == pytest.approx(0.68)


def test_text_in_the_middle_of_the_frame_collapses_the_zone_and_that_is_the_finding():
    """No edge cut removes a centred logo without taking most of the picture, and
    smoothing that over would be the app claiming a clip is croppable when it is
    not. A small `safe_area` is the honest answer to "can I save this one"."""
    rect = geo.safe_rect(None, [(0.40, 0.42, 0.60, 0.58)])
    assert geo.safe_area(rect) < 0.45
    assert geo.safe_area(rect) > 0.0


def test_a_zone_that_covers_everything_leaves_nothing_rather_than_a_negative():
    rect = geo.safe_rect(None, [(0.0, 0.0, 1.0, 1.0)])
    assert rect == (0.0, 0.0, 0.0, 0.0)
    assert geo.safe_area(rect) == 0.0


def test_the_zone_is_stable_across_runs_for_the_same_boxes():
    """A stored measurement that moved between identical runs would read as
    footage that changed. Ties are broken on the candidate itself, never on set
    or dict ordering."""
    boxes = [(0.0, 0.0, 0.3, 0.3), (0.7, 0.7, 1.0, 1.0)]
    first = geo.safe_rect(None, boxes)
    assert all(geo.safe_rect(None, list(reversed(boxes))) == first
               for _ in range(3))


# --- the cuts ------------------------------------------------------------------------

def test_the_three_cuts_are_ones_the_panel_and_the_route_both_know():
    """THE canonical list. A cut that exists only in `verdicts()` is settable by
    hand-editing config.json and by nothing else — the failure
    `first_frame_floor` shipped with."""
    for key in ('bars_max', 'text_coverage_max', 'safe_area_min'):
        assert key in video_metrics.THRESHOLD_KEYS


def test_the_cuts_ship_with_no_number():
    """Bands and burned text describe FOOTAGE, not a classifier's calibrated
    probability — so, like every footage cut in this lane, they ship empty and
    the published references live in the hints."""
    from app.config import DEFAULTS
    for key in ('bars_max', 'text_coverage_max', 'safe_area_min'):
        assert DEFAULTS['video_bank'][key] is None


def test_the_threshold_reader_hands_the_cuts_through(app):
    from app.services import video_bank_service as svc
    with app.app_context():
        reader = svc.metric_thresholds()
    for key in ('bars_max', 'text_coverage_max', 'safe_area_min'):
        assert key in reader


def test_each_cut_raises_its_own_flag():
    assert 'letterboxed' in video_metrics.verdicts({'bars_ratio': 0.20},
                                                   {'bars_max': 0.04})
    assert 'burned_text' in video_metrics.verdicts({'text_coverage': 0.03},
                                                   {'text_coverage_max': 0.01})
    assert 'small_safe_zone' in video_metrics.verdicts({'safe_area': 0.35},
                                                       {'safe_area_min': 0.6})


def test_a_shot_inside_the_cuts_raises_nothing():
    flags = video_metrics.verdicts(
        {'bars_ratio': 0.01, 'text_coverage': 0.0, 'safe_area': 0.99},
        {'bars_max': 0.04, 'text_coverage_max': 0.01, 'safe_area_min': 0.6})
    assert flags == set()


def test_an_unmeasured_shot_is_never_flagged_whatever_the_cuts():
    """"Not evaluated" must not read as "letterboxed". A bank measured before
    this shipped would otherwise report all of itself as its worst footage."""
    assert video_metrics.verdicts({'metrics_state': 'ok'},
                                  {'bars_max': 0.0, 'text_coverage_max': 0.0,
                                   'safe_area_min': 1.0}) == set()


def test_a_bands_only_shot_can_be_flagged_for_bands_and_never_for_text():
    """The degraded install, end to end at read time: the OCR engine was absent,
    so the bands ARE a measurement and the text is not one."""
    stored = {'safe_zone_state': 'bars_only', 'bars_ratio': 0.24}
    cuts = {'bars_max': 0.04, 'text_coverage_max': 0.0, 'safe_area_min': 1.0}
    assert video_metrics.verdicts(stored, cuts) == {'letterboxed'}


# --- the pass over a bank -------------------------------------------------------------

def test_the_pass_measures_bands_and_text_on_every_shot(app, monkeypatch):
    bank_id, ids = _bank_with_clips(app, 2)
    _fake_frames(monkeypatch, bands=_letterboxed(100, 100, 12))
    _fake_text(monkeypatch, {'*': [SUB_A]})

    with app.app_context():
        out = sz.run_safe_zone(bank_id)

    assert out['measured'] == 2
    assert out['unreadable'] == 0
    assert out['letterboxed'] == 2
    assert out['error'] is None
    stored = _summaries(app, bank_id)[ids[0]]
    assert stored['safe_zone_state'] == 'ok'
    assert stored['safe_zone_frames'] == 3
    assert stored['safe_bands'] == [0.12, 0.12, 0.0, 0.0]
    assert stored['bars_ratio'] == pytest.approx(0.24)
    # 0.6 wide x 0.08 tall, and the subtitle is inside the picture area.
    assert stored['text_coverage'] == pytest.approx(0.048, abs=1e-3)
    assert stored['safe_rect'] == [0.0, 0.12, 1.0, 0.82]
    assert stored['safe_area'] == pytest.approx(0.70, abs=1e-3)


def test_one_decode_feeds_both_halves(app, monkeypatch):
    """The bands and the OCR read the SAME frames, written ONCE.

    Decoding is the expensive part of this feature — three frames per shot at
    768 px — and splitting it into "a band pass" and "a text pass" is the
    obvious-looking refactor that would double it for no gain. So: exactly one
    `_write_frames` call per shot, and every path handed to the OCR child is one
    of the paths the bands were measured on. Not a byte more decoded, whatever
    the install can do with the result."""
    bank_id, ids = _bank_with_clips(app, 2)
    calls = []
    banded = []

    def _write(src, times, dest, stem, long_side=None):
        calls.append((stem, long_side, [label for label, _s in times]))
        return [(label, seconds, f'{dest}/{stem}_{label}.jpg')
                for label, seconds in times]

    monkeypatch.setattr(sz, '_write_frames', _write)
    monkeypatch.setattr(sz, 'luma_grid',
                        lambda path, probe=None: banded.append(path)
                        or _letterboxed(100, 100, 12))
    seen_by_ocr = []
    monkeypatch.setattr(sz, 'text_engine_reason', lambda: None)
    monkeypatch.setattr(
        sz, 'read_text_boxes',
        lambda frames, **kw: (seen_by_ocr.extend(f['path'] for f in frames)
                              or {f['key']: [] for f in frames}))

    with app.app_context():
        assert sz.run_safe_zone(bank_id)['measured'] == 2

    # ONE extraction per shot, at the resolution this pass chose.
    assert [stem for stem, _ls, _labels in calls] == [f'clip_{c}' for c in ids]
    assert {long_side for _stem, long_side, _labels in calls} == {sz.FRAME_LONG_SIDE}
    # The OCR saw exactly the frames the bands were measured on — same paths,
    # same count, no second decode hiding anywhere.
    assert sorted(seen_by_ocr) == sorted(banded)
    assert len(seen_by_ocr) == 6                    # 2 shots x 3 frames


def test_a_shot_whose_frames_cannot_be_decoded_is_unreadable_and_never_zero(
        app, monkeypatch):
    """Zeros here would be the app asserting the frame is entirely usable AND
    entirely letterboxed at once. The state is the answer."""
    bank_id, ids = _bank_with_clips(app, 1)
    monkeypatch.setattr(sz, '_write_frames',
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError('no frame could be decoded')))
    _fake_text(monkeypatch, {})

    with app.app_context():
        out = sz.run_safe_zone(bank_id)

    assert out == {'measured': 0, 'letterboxed': 0, 'unreadable': 1,
                   'text_frames': 0, 'error': None}
    stored = _summaries(app, bank_id)[ids[0]]
    assert stored == {'safe_zone_state': 'unreadable'}
    assert 'bars_ratio' not in stored
    assert 'text_coverage' not in stored


def test_with_no_ocr_engine_the_bands_are_still_measured_and_the_text_is_not(
        app, monkeypatch):
    """The install this pass is designed around. Half of it needs nothing, so a
    missing extra downgrades the result and says so — it does not turn the pass
    off, and it does not write a zero coverage nobody measured."""
    bank_id, ids = _bank_with_clips(app, 1)
    _fake_frames(monkeypatch, bands=_letterboxed(100, 100, 12))
    monkeypatch.setattr(sz, 'text_engine_reason',
                        lambda: 'burned-in text was not read — install it from Setup')
    monkeypatch.setattr(sz, 'read_text_boxes',
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError('the OCR child was started anyway')))

    with app.app_context():
        out = sz.run_safe_zone(bank_id)

    assert out['measured'] == 1
    assert 'install it from Setup' in out['error']
    stored = _summaries(app, bank_id)[ids[0]]
    assert stored['safe_zone_state'] == 'bars_only'
    assert stored['bars_ratio'] == pytest.approx(0.24)
    for key in ('text_coverage', 'safe_rect', 'safe_area'):
        assert key not in stored, \
            f'{key} was written without the text ever being read'


def test_a_shot_the_text_reader_never_reached_stays_in_the_queue(app, monkeypatch):
    """A Stop, or a budget that elapsed, mid-chunk. Storing its bands alone would
    retire it as `bars_only` on an install that reads text perfectly well, and
    storing a zero coverage would be worse."""
    bank_id, ids = _bank_with_clips(app, 2)
    _fake_frames(monkeypatch, bands=_noise(100, 100))
    # Only the first clip's frames come back.
    monkeypatch.setattr(sz, 'text_engine_reason', lambda: None)
    monkeypatch.setattr(
        sz, 'read_text_boxes',
        lambda frames, **kw: {f['key']: [] for f in frames
                              if f['key'].startswith(f'{ids[0]}:')})

    with app.app_context():
        out = sz.run_safe_zone(bank_id)
        assert out['measured'] == 1
        assert [c.id for c in sz.pending_clips(bank_id)] == [ids[1]]
    assert _summaries(app, bank_id)[ids[1]] == {}


def test_a_frame_read_with_no_text_is_a_measurement_and_not_a_gap(app, monkeypatch):
    """The other half of the rule above: an EMPTY box list is an answer. A shot
    whose frames were all read and hold nothing must be retired as measured, or
    a clean bank would be re-scanned forever."""
    bank_id, ids = _bank_with_clips(app, 1)
    _fake_frames(monkeypatch, bands=_noise(100, 100))
    _fake_text(monkeypatch, {'*': []})

    with app.app_context():
        out = sz.run_safe_zone(bank_id)
        assert out['measured'] == 1
        assert sz.pending_clips(bank_id) == []
    stored = _summaries(app, bank_id)[ids[0]]
    assert stored['text_coverage'] == 0.0
    assert stored['safe_area'] == 1.0


def test_a_shot_whose_every_frame_is_a_fade_claims_no_usable_frame_either(
        app, monkeypatch):
    """Three uniform frames — a fade, a black slug, a title card. The container
    is then UNKNOWN, so a `safe_area: 1.0` would be the app announcing that the
    whole of a black frame is usable. The text WAS measured and stays; the rest
    is absent, and `safe_zone_frames: 0` says why."""
    bank_id, ids = _bank_with_clips(app, 1)
    _fake_frames(monkeypatch, bands=_grid(100, 100, fill=0))
    _fake_text(monkeypatch, {'*': []})

    with app.app_context():
        assert sz.run_safe_zone(bank_id)['measured'] == 1

    stored = _summaries(app, bank_id)[ids[0]]
    assert stored['safe_zone_state'] == 'ok'
    assert stored['safe_zone_frames'] == 0
    assert stored['text_coverage'] == 0.0
    for key in ('safe_bands', 'bars_ratio', 'safe_rect', 'safe_area'):
        assert key not in stored, f'{key} was written about a container nobody saw'
    assert video_metrics.verdicts(stored, {'bars_max': 0.0,
                                           'safe_area_min': 1.0}) == set()


def test_a_reader_that_dies_leaves_the_shots_in_the_queue_rather_than_bars_only(
        app, monkeypatch):
    """The two reasons text can go unread do NOT get the same answer.

    'bars_only' is a claim about the INSTALL — "there is no OCR engine here" —
    and a rescan after installing one is what fixes it. A child that died on a
    machine that HAS the engine (an OOM, a killed process) is a different fact
    with a different fix, and writing `bars_only` for it would retire those
    shots forever on an install that reads text perfectly well. So the chunk is
    abandoned whole and its shots keep no state at all.

    Everything committed BEFORE the failure stays, and the sentence survives to
    the caller — that half is the watermark pass's contract, kept."""
    bank_id, ids = _bank_with_clips(app, 1)
    _fake_frames(monkeypatch, bands=_letterboxed(100, 100, 12))
    monkeypatch.setattr(sz, 'text_engine_reason', lambda: None)
    monkeypatch.setattr(sz, 'read_text_boxes',
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError('the text reader produced no result')))

    with app.app_context():
        out = sz.run_safe_zone(bank_id)
        assert out['measured'] == 0
        assert 'produced no result' in out['error']
        assert [c.id for c in sz.pending_clips(bank_id)] == [ids[0]]
    assert _summaries(app, bank_id)[ids[0]] == {}


def test_a_second_run_measures_nothing_and_never_starts_the_reader(app, monkeypatch):
    """The resume contract: re-clicking on a measured bank costs nothing."""
    bank_id, _ids = _bank_with_clips(app, 1)
    _fake_frames(monkeypatch, bands=_noise(100, 100))
    _fake_text(monkeypatch, {'*': []})
    with app.app_context():
        sz.run_safe_zone(bank_id)

    monkeypatch.setattr(sz, '_write_frames',
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError('a measured bank was decoded again')))
    with app.app_context():
        assert sz.run_safe_zone(bank_id) == {'measured': 0, 'letterboxed': 0,
                                             'unreadable': 0, 'text_frames': 0,
                                             'error': None}


def test_a_rescan_measures_the_whole_bank_again(app, monkeypatch):
    bank_id, ids = _bank_with_clips(app, 1)
    _fake_frames(monkeypatch, bands=_noise(100, 100))
    _fake_text(monkeypatch, {'*': []})
    with app.app_context():
        sz.run_safe_zone(bank_id)

    _fake_frames(monkeypatch, bands=_letterboxed(100, 100, 20))
    with app.app_context():
        assert sz.run_safe_zone(bank_id, rescan=True)['measured'] == 1
    assert _summaries(app, bank_id)[ids[0]]['bars_ratio'] == pytest.approx(0.40)


def test_a_rescan_without_the_engine_does_not_leave_the_old_text_numbers(
        app, monkeypatch):
    """The reason `_store` clears this pass's own keys before writing. A
    `safe_area` from a run that could read text, sitting beside a `bars_only`
    state, is a number nobody can interpret."""
    bank_id, ids = _bank_with_clips(app, 1)
    _fake_frames(monkeypatch, bands=_noise(100, 100))
    _fake_text(monkeypatch, {'*': [SUB_A]})
    with app.app_context():
        sz.run_safe_zone(bank_id)
    assert 'safe_area' in _summaries(app, bank_id)[ids[0]]

    monkeypatch.setattr(sz, 'text_engine_reason', lambda: 'no OCR engine here')
    with app.app_context():
        sz.run_safe_zone(bank_id, rescan=True)
    stored = _summaries(app, bank_id)[ids[0]]
    assert stored['safe_zone_state'] == 'bars_only'
    assert 'safe_area' not in stored
    assert 'text_coverage' not in stored


def test_the_pass_keeps_the_measurements_it_did_not_write(app, monkeypatch):
    bank_id, ids = _bank_with_clips(app, 1)
    _measured(app, {ids[0]: 12.0})
    _fake_frames(monkeypatch, bands=_noise(100, 100))
    _fake_text(monkeypatch, {'*': []})

    with app.app_context():
        sz.run_safe_zone(bank_id)

    stored = _summaries(app, bank_id)[ids[0]]
    assert stored['sharpness_p90'] == 12.0
    assert stored['metrics_state'] == 'ok'
    assert stored['safe_zone_state'] == 'ok'


def test_the_pass_never_changes_a_triage_decision(app, monkeypatch):
    """Advisory means advisory: this writes numbers, it does not reject."""
    bank_id, _ids = _bank_with_clips(app, 2)
    _fake_frames(monkeypatch, bands=_letterboxed(100, 100, 40))
    _fake_text(monkeypatch, {'*': [(0.0, 0.0, 1.0, 1.0)]})

    with app.app_context():
        from app.models import VideoClip
        sz.run_safe_zone(bank_id)
        assert {c.status for c in VideoClip.query.filter_by(bank_id=bank_id)} \
            == {'pending'}


def test_moving_a_cut_re_sorts_the_bank_with_nothing_re_scanned(app, monkeypatch):
    """The doctrine of the whole lane. The blob is read once and asked twice —
    the flag appears and disappears with the cut, and the stored bytes never
    move."""
    bank_id, ids = _bank_with_clips(app, 1)
    _fake_frames(monkeypatch, bands=_letterboxed(100, 100, 12))
    _fake_text(monkeypatch, {'*': [SUB_A]})
    with app.app_context():
        sz.run_safe_zone(bank_id)

    stored = _summaries(app, bank_id)[ids[0]]
    before = json.dumps(stored, sort_keys=True)
    assert 'letterboxed' in video_metrics.verdicts(stored, {'bars_max': 0.10})
    assert 'letterboxed' not in video_metrics.verdicts(stored, {'bars_max': 0.30})
    assert 'burned_text' in video_metrics.verdicts(stored,
                                                   {'text_coverage_max': 0.01})
    assert 'burned_text' not in video_metrics.verdicts(stored,
                                                       {'text_coverage_max': 0.10})
    assert 'small_safe_zone' in video_metrics.verdicts(stored,
                                                       {'safe_area_min': 0.80})
    assert 'small_safe_zone' not in video_metrics.verdicts(stored,
                                                           {'safe_area_min': 0.60})
    assert json.dumps(_summaries(app, bank_id)[ids[0]], sort_keys=True) == before


def test_a_shot_of_an_unprobed_source_is_never_queued(app, monkeypatch):
    """An unreadable file has no frame to extract, and counting it as unreadable
    on every run would make the pass look permanently broken."""
    bank_id, _ids = _bank_with_clips(app, 1, probe_state='error')
    with app.app_context():
        assert sz.pending_clips(bank_id) == []


# --- the shared blob ------------------------------------------------------------------

def test_every_key_this_pass_owns_survives_a_re_measure():
    """The metrics scan rewrites metrics_json wholesale and knows nothing about
    OCR — it could not recompute any of this. A key missing from ADVISORY_KEYS is
    erased by the next 📊 Measure with nothing anywhere to see."""
    missing = [k for k in sz.OWNED_KEYS if k not in video_metrics.ADVISORY_KEYS]
    assert not missing, (
        f'{missing} would be wiped by a re-measure — add them to '
        'video_metrics.ADVISORY_KEYS')

    stored = {k: 'x' for k in sz.OWNED_KEYS}
    stored['metrics_state'] = 'ok'
    merged = video_metrics.merge_advisory(stored, {'metrics_state': 'ok',
                                                   'sharpness_p90': 9.0})
    for key in sz.OWNED_KEYS:
        assert merged[key] == 'x'
    assert merged['sharpness_p90'] == 9.0


# --- the wiring -----------------------------------------------------------------------

def test_the_pass_has_a_button_that_reaches_it(app, monkeypatch):
    """A feature nobody can start is dead code. The job kind is what the labels,
    the progress line and the 409 all key on, so it is pinned here."""
    from app.services import bank_jobs
    from app.services import video_bank_service as svc
    bank_id, _ids = _bank_with_clips(app, 1)
    started = {}
    monkeypatch.setattr(bank_jobs, 'start',
                        lambda app_, key, kind, fn: started.update(kind=kind, fn=fn))
    with app.app_context():
        svc.start_safe_zone(app, 'local', bank_id)
    assert started['kind'] == 'safezone'


def test_the_route_is_mounted_and_refuses_only_a_missing_decoder(app, monkeypatch):
    """ONE 503, unlike 🔖 Watermarks' two: with no decoder there is no frame to
    look at, but with no OCR engine half the pass still works and must not be
    withheld behind a status code."""
    from app import capabilities
    bank_id, _ids = _bank_with_clips(app, 1)
    client = app.test_client()

    monkeypatch.setattr(capabilities, 'probe_video',
                        lambda: {'decode': False, 'detail': 'av is missing'})
    assert client.post(f'/api/video-bank/{bank_id}/safezone').status_code == 503

    monkeypatch.setattr(capabilities, 'probe_video',
                        lambda: {'decode': True, 'detail': 'ok'})
    monkeypatch.setattr('app.services.video_bank_service.start_safe_zone',
                        lambda *a, **k: None)
    assert client.post(f'/api/video-bank/{bank_id}/safezone').status_code == 202


def test_the_job_says_what_it_could_not_do_instead_of_reporting_a_clean_bank(
        app, monkeypatch):
    """A run with no OCR engine finishes successfully and finds no text. Silence
    there leaves a bank whose text cut flags nothing, with nothing to explain
    why."""
    from app.services import video_bank_service as svc
    bank_id, _ids = _bank_with_clips(app, 1)
    _fake_frames(monkeypatch, bands=_letterboxed(100, 100, 12))
    monkeypatch.setattr(sz, 'text_engine_reason', lambda: 'RapidOCR is not installed')

    job = {'done': 0, 'total': 1, 'detail': ''}
    with app.app_context():
        out = svc._safe_zone_job(bank_id, False)(job)

    assert out['measured'] == 1
    assert 'RapidOCR is not installed' in job['detail']
    assert '1 with bands' in job['detail']


def test_the_capability_the_workspace_reads_rides_the_bank_payload(app, monkeypatch):
    """The button never greys out for a missing OCR engine — it says "bands
    only" instead — and it can only do that if the payload carries the answer."""
    from app import capabilities
    from app.services import video_bank_service as svc
    monkeypatch.setattr(capabilities, 'probe_video_text',
                        lambda: {'ok': False, 'detail': 'install it from Setup'})
    with app.app_context():
        cap = svc._capability()
    assert cap['video_text'] is False
    assert cap['video_text_detail'] == 'install it from Setup'


# --- the worker's own contracts --------------------------------------------------------

def test_the_worker_never_takes_the_card():
    """A bank must be measurable while a training run owns the GPU — that is the
    whole reason the text half is onnxruntime rather than the torch detector this
    lane already ships."""
    src = _infer_source('video_text_infer.py')
    hide = src.index("os.environ['CUDA_VISIBLE_DEVICES'] = ''")
    assert hide < src.index('import cv2'), \
        'CUDA must be hidden before anything heavy is imported'


def test_the_probe_imports_everything_the_worker_imports():
    """Issue #24's exact shape: a probe naming only the headline module reports
    ✓ while the feature dies on the first call. `cv2` arrives as a DEPENDENCY of
    RapidOCR rather than in its own right, which is precisely the case that
    broke `masks`."""
    import re
    from app import capabilities
    src = _infer_source('video_text_infer.py')
    probe = capabilities.CAPABILITY_IMPORTS['video_text']
    # Read off the worker's own import statements rather than a hand-written
    # list, so a THIRD dependency added there fails this test instead of shipping
    # a probe that under-reports.
    imported = set(re.findall(r'^\s*(?:import|from)\s+([A-Za-z_][\w]*)', src,
                              re.M))
    # The stdlib is always there; _harness is the stdlib-only sibling module
    # shipped NEXT TO the worker (same directory, same repo), so it can no more
    # be absent from the env than json can — the probe owes it nothing.
    imported -= {'__future__', 'json', 'os', 'sys', '_harness'}
    assert imported == {'rapidocr_onnxruntime', 'cv2', 'numpy', 'PIL'}, \
        f'the worker imports {sorted(imported)} — update the probe to match'
    for module in imported:
        assert module in probe, \
            f'the worker imports {module} and the capability probe does not'


def test_the_worker_normalises_its_boxes_and_drops_low_confidence_ones():
    """Run OUT OF PROCESS on purpose: importing the worker sets
    CUDA_VISIBLE_DEVICES for every test after it. Nothing here needs RapidOCR —
    only the quad→rectangle conversion, which is ours."""
    code = (
        'import json, sys\n'
        f'sys.path.insert(0, {str(_infer_dir())!r})\n'
        'import video_text_infer as w\n'
        'quad = [[100, 50], [300, 50], [300, 90], [100, 90]]\n'
        'print(json.dumps({\n'
        '  "kept": w._boxes_of([(quad, "HELLO", 0.9)], 400, 200, 0.5),\n'
        '  "dropped": w._boxes_of([(quad, "HELLO", 0.2)], 400, 200, 0.5),\n'
        '  "rotated": w._boxes_of('
        '      [([[10, 20], [90, 10], [95, 40], [15, 50]], "X", 0.8)], 100, 100, 0.5),\n'
        '  "junk": w._boxes_of([("not a quad",)], 400, 200, 0.5),\n'
        '}))\n')
    proc = subprocess.run([sys.executable, '-c', code], capture_output=True,
                          text=True, encoding='utf-8', errors='replace')
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    # 100..300 of 400 wide, 50..90 of 200 tall.
    assert out['kept'] == [[0.25, 0.25, 0.75, 0.45, 0.9]]
    assert out['dropped'] == []
    # A rotated quad becomes its bounding box — the conservative direction for a
    # measurement that justifies a crop.
    assert out['rotated'] == [[0.1, 0.1, 0.95, 0.5, 0.8]]
    assert out['junk'] == []


def test_the_text_extra_can_be_installed_from_setup():
    """CLAUDE.md's rule, checked here as well as by the global contract test:
    the pass is the only thing in the app that DEGRADES on a missing capability
    instead of refusing, and a degraded state with no repair button would be a
    permanent one."""
    from app import setup_installer
    assert 'video_text' in setup_installer.INSTALL_ACTIONS
    packages = setup_installer._CAPABILITY_PACKAGES['video_text']
    assert 'rapidocr-onnxruntime' in packages
    # The headless cv2, like face_scoring and masks: RapidOCR depends on the
    # DESKTOP opencv-python, which drags a GUI stack onto a server.
    assert 'opencv-python-headless' in packages
    req = (Path(__file__).resolve().parents[1]
           / 'requirements-ml.txt').read_text(encoding='utf-8')
    assert 'rapidocr-onnxruntime>=' in req, 'the version floor is not pinned'
    assert '<2' in req.split('rapidocr-onnxruntime')[1].splitlines()[0], \
        ('RapidOCR 2.x renamed the import and moved the weights to a download — '
         'the ceiling is what keeps the worker importable and offline-capable')


# --- helpers ---------------------------------------------------------------------------

def _infer_dir():
    return Path(__file__).resolve().parents[1] / 'infer'


def _infer_source(name):
    return (_infer_dir() / name).read_text(encoding='utf-8')


def _bank_with_clips(app, n, probe_state='ok'):
    from app.extensions import db
    from app.models import VideoBank, VideoClip, VideoSource
    with app.app_context():
        bank = VideoBank(name='b', source_path='/srv/rushes')
        db.session.add(bank)
        db.session.flush()
        src = VideoSource(bank_id=bank.id, relpath='a.mp4', duration_s=60.0,
                          fps_native=25.0, probe_state=probe_state)
        db.session.add(src)
        db.session.flush()
        ids = []
        for i in range(n):
            clip = VideoClip(bank_id=bank.id, source_id=src.id,
                             start_s=float(i * 10), end_s=float(i * 10 + 5))
            db.session.add(clip)
            db.session.flush()
            ids.append(clip.id)
        db.session.commit()
        return bank.id, ids


def _fake_frames(monkeypatch, *, bands):
    """The DECODE seam, stubbed: three named frames per shot, all reading as the
    same forged luma grid. Nothing here opens a video."""
    monkeypatch.setattr(
        sz, '_write_frames',
        lambda src, times, dest, stem, long_side=None: [
            (label, seconds, f'{dest}/{stem}_{label}.jpg')
            for label, seconds in times])
    monkeypatch.setattr(sz, 'luma_grid', lambda path, probe=None: bands)


def _fake_text(monkeypatch, per_key):
    """The MODEL seam, stubbed. `{'*': boxes}` answers the same boxes for every
    frame; any other key matches one frame exactly."""
    monkeypatch.setattr(sz, 'text_engine_reason', lambda: None)

    def _read(frames, **kw):
        return {f['key']: [list(b) + [0.9] for b in
                           per_key.get(f['key'], per_key.get('*', []))]
                for f in frames}

    monkeypatch.setattr(sz, 'read_text_boxes', _read)


def _measured(app, sharpness_by_id):
    from app.extensions import db
    from app.models import VideoClip
    with app.app_context():
        for cid, sharp in sharpness_by_id.items():
            db.session.get(VideoClip, cid).metrics_json = json.dumps(
                {'metrics_state': 'ok', 'sharpness_p90': sharp})
        db.session.commit()


def _summaries(app, bank_id):
    from app.models import VideoClip
    with app.app_context():
        rows = VideoClip.query.filter_by(bank_id=bank_id).all()
        return {r.id: (json.loads(r.metrics_json) if r.metrics_json else {})
                for r in rows}
