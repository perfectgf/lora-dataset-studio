"""🤖 The AI check — the statistic, its polarity, and everything it refuses to claim.

What is under test is NOT "can a transformer tell real from generated" (that is
somebody else's checkpoint and it cannot be imported in this interpreter) but the
decisions that make a hedged, ~three-in-four verdict safe to put in front of a
user:

  * the REDUCTION ORDER, which the paper's prose invites you to get wrong — the
    feature VECTOR is collapsed to a scalar distance BEFORE any differencing, so
    the statistic runs over a 1-D series and not a 768-d one;
  * the POLARITY, which is inverted against every other cut in the panel: a LOW
    score is the suspicious one, so the cut is a FLOOR and raising it flags more;
  * ONE WINDOW, always the same size, because the standard deviation's scale
    depends on how many samples it spans — a partial window is NO measurement
    rather than a smaller one;
  * every degraded case is a STATE and never a zero: too short for the window,
    frames unreadable, or skipped by the child (which keeps NO key at all, and
    that absence is what puts the shot back in the queue);
  * the verdict is derived at READ time, so moving the cut re-sorts the bank with
    nothing rescanned;
  * the port is FAITHFUL — the preprocessing quirks are reproduced on purpose and
    pinned here, because "fixing" them silently would take the feature off the
    only configuration anyone has measured.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import config as cfg
from app.services import video_ai_check as ac
from app.services import video_metrics


def _infer_dir():
    return Path(__file__).resolve().parents[1] / 'infer'


def _infer_source(name):
    return (_infer_dir() / name).read_text(encoding='utf-8')


def _infer_code(name):
    """The worker's SOURCE with its module docstring removed.

    The docstring names the alternatives it deliberately does NOT use — CLIP's
    own normalisation constants, albumentations' `Normalize` — so a naive search
    over the whole file finds them and calls the port broken. What has to be
    checked is what the code DOES."""
    src = _infer_source(name)
    return src.split('"""', 2)[2]


def _infer_module():
    """The worker, imported for its pure halves. It imports torch and
    transformers INSIDE main(), which is what makes this possible at all."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'video_ai_check_infer', _infer_dir() / 'video_ai_check_infer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- the mathematical core, on forged features ------------------------------------

def test_step_distances_are_l2_between_consecutive_feature_vectors():
    """The FIRST half of the statistic, on features whose answer is arithmetic.

    Three 2-d vectors one unit apart, then three apart: the steps are the
    distances between neighbours and nothing else — no normalisation of the
    features (which would flatten the very magnitude this measures) and no
    differencing yet."""
    pytest.importorskip('numpy')
    infer = _infer_module()
    features = [[0.0, 0.0], [1.0, 0.0], [4.0, 0.0], [4.0, 4.0]]
    assert infer.step_distances(features) == pytest.approx([1.0, 3.0, 4.0])


def test_step_distances_needs_two_frames_and_returns_nothing_below_that():
    pytest.importorskip('numpy')
    infer = _infer_module()
    assert infer.step_distances([[1.0, 2.0]]) == []
    assert infer.step_distances([]) == []


def test_a_perfectly_regular_signal_scores_zero_and_a_noisy_one_does_not():
    """THE POLARITY, on the two shapes the feature exists to separate.

    A subject moving at a perfectly constant speed produces constant steps, so
    the differences between them are all zero and the spread of those is zero.
    Anything that accelerates unevenly — which is everything real — produces a
    positive one. Low is the suspicious side, and that is what the floor cuts on.
    """
    smooth = [2.0] * 16
    noisy = [2.0, 3.5, 1.2, 4.1, 2.0, 3.9, 1.5, 3.0, 2.2, 4.4, 1.1, 3.3, 2.6,
             3.8, 1.4, 2.9]
    assert ac.irregularity(smooth) == pytest.approx(0.0)
    assert ac.irregularity(noisy) > 1.0
    # The direction, stated as the product claim rather than as two numbers: the
    # generated-looking clip is BELOW the filmed-looking one. An implementation
    # that inverted the statistic would still pass both assertions above.
    assert ac.irregularity(smooth) < ac.irregularity(noisy)


def test_a_steadily_accelerating_signal_still_scores_zero():
    """The subtle half of "second order". A subject speeding up smoothly has
    steps that GROW — a first-order statistic would call it irregular — but the
    growth itself is constant, so the second differences are identical and the
    spread is zero. This is the whole reason the method differences twice."""
    steps = [1.0 + 0.5 * i for i in range(16)]
    assert ac.irregularity(steps) == pytest.approx(0.0)


def test_the_spread_is_bessel_corrected_like_the_reference():
    """Hand-computed, because an n vs n-1 denominator is invisible in a diff and
    would silently rescale every score in every bank.

    steps [0, 1, 3, 6] -> second differences [1, 2, 3] -> mean 2, deviations
    [-1, 0, 1] -> sum of squares 2 -> /(3-1) = 1 -> sqrt = 1. The population form
    would divide by 3 and answer 0.8165."""
    assert ac.irregularity([0.0, 1.0, 3.0, 6.0]) == pytest.approx(1.0)
    assert ac.irregularity([0.0, 1.0, 3.0, 6.0]) != pytest.approx(0.8165, abs=1e-3)


def test_too_few_steps_is_no_measurement_and_never_a_zero():
    """A series with fewer than two second differences has no spread to report.
    None, not 0.0 — a zero here is the strongest possible claim that a shot is
    synthetic, made about a shot nothing measured."""
    assert ac.irregularity([]) is None
    assert ac.irregularity([1.0]) is None
    assert ac.irregularity([1.0, 2.0]) is None
    assert ac.irregularity([1.0, 2.0, 4.0]) is not None


def test_the_score_is_not_normalised_by_how_much_the_shot_moves():
    """The port stays a port. Doubling every step doubles the score, because the
    statistic is an ABSOLUTE standard deviation and not a coefficient of
    variation. Measured on this app's own forged clips, the normalised variant
    gives the identical AUC and the identical ordering, so there is no reason to
    deviate — and a silent switch to a relative form would move every stored
    number and every cut set against it."""
    steps = [1.0, 2.0, 1.5, 3.0, 2.5, 1.0, 2.0]
    doubled = [2 * s for s in steps]
    assert ac.irregularity(doubled) == pytest.approx(2 * ac.irregularity(steps))


# --- the window ---------------------------------------------------------------------

def test_the_window_is_sixteen_contiguous_instants_at_eight_frames_a_second():
    times = ac.window_times(10.0, 20.0)
    assert len(times) == ac.FRAMES == 16
    gaps = [round(times[i + 1] - times[i], 6) for i in range(len(times) - 1)]
    assert gaps == [pytest.approx(0.125)] * 15


def test_the_window_is_centred_so_both_cuts_keep_their_margin():
    """A dissolve lives at a boundary and is violently irregular — it would push
    a score UP and hide a generated clip. Head-anchoring protects one boundary;
    centring protects both, and puts the sample where the shot is most itself."""
    times = ac.window_times(10.0, 20.0)
    head = times[0] - 10.0
    tail = 20.0 - times[-1]
    assert head == pytest.approx(tail)
    assert head >= ac.EDGE_MARGIN_S


def test_a_shot_too_short_for_the_window_yields_no_times_at_all():
    """The minimum is the window plus a margin at each end. Just under it must
    produce NOTHING rather than a squeezed window — a shorter sample is a
    different estimator on a different scale, and one cut across two scales
    would flag shots for their length."""
    floor = ac.min_duration_s()
    assert floor == pytest.approx((16 - 1) / 8.0 + 2 * 0.25)
    assert ac.window_times(0.0, floor - 0.01) == []
    assert len(ac.window_times(0.0, floor + 0.01)) == ac.FRAMES


def test_the_frame_count_is_a_multiple_of_the_encoders_group_size():
    """XCLIP passes messages between frames in groups of `num_frames` (8 for
    this checkpoint). A count that is not a multiple of it raises inside the
    model — which is why the reference hardcodes 8-or-16 — and a count that
    straddles a group boundary would mix two clips' motion."""
    assert ac.FRAMES % 8 == 0


def test_the_centre_crop_takes_the_long_edge_and_leaves_the_short_one_whole():
    """The reference's own geometry: `fraction` off EACH END of the longer edge,
    the shorter edge untouched. Note the paper says 10 % of the longer edge while
    its code removes 20 % of it — the CODE produced the published numbers."""
    assert ac.crop_box(1000, 500, 0.1) == (100, 0, 900, 500)
    assert ac.crop_box(500, 1000, 0.1) == (0, 100, 500, 900)
    # Square: the width branch wins, and either is equally defensible — what
    # matters is that it is deterministic.
    assert ac.crop_box(400, 400, 0.1) == (40, 0, 360, 400)


def test_a_degenerate_frame_size_returns_the_whole_frame_rather_than_an_empty_box():
    """An empty crop would raise inside the decode seam and retire a shot as
    unreadable for a reason that has nothing to do with the shot."""
    assert ac.crop_box(2, 2, 0.1) == (0, 0, 2, 2)
    assert ac.crop_box(0, 0, 0.1) == (0, 0, 0, 0)
    assert ac.crop_box(100, 60, 0.5) == (0, 0, 100, 60)


# --- the pass, with the model seam monkeypatched --------------------------------------

def _bank_with_clips(app, spans, probe_state='ok'):
    from app.extensions import db
    from app.models import VideoBank, VideoClip, VideoSource
    with app.app_context():
        bank = VideoBank(name='b', source_path='/srv/rushes',
                         user_id=cfg.LOCAL_USER)
        db.session.add(bank)
        db.session.flush()
        src = VideoSource(bank_id=bank.id, relpath='a.mp4', duration_s=600.0,
                          fps_native=25.0, probe_state=probe_state)
        db.session.add(src)
        db.session.flush()
        ids = []
        for start, end in spans:
            clip = VideoClip(bank_id=bank.id, source_id=src.id,
                             start_s=float(start), end_s=float(end))
            db.session.add(clip)
            db.session.flush()
            ids.append(clip.id)
        db.session.commit()
        return bank.id, ids


def _long(n):
    """`n` shots comfortably longer than the window."""
    return [(i * 20.0, i * 20.0 + 10.0) for i in range(n)]


def _stub_decode(monkeypatch):
    """Every window decodes, with FRAMES fake paths and no PyAV anywhere."""
    monkeypatch.setattr(ac, '_write_window',
                        lambda path, times, dest, stem: [f'{stem}_{i}.jpg'
                                                         for i in range(len(times))])


def _stub_model(monkeypatch, steps_for):
    """`steps_for(clip_id)` -> the step series the child would return, or None to
    have the child skip that clip entirely."""
    def fake(payload, *, timeout=None):
        out = {}
        for item in payload:
            series = steps_for(item['id'])
            if series is not None:
                out[item['id']] = series
        return out
    monkeypatch.setattr(ac, 'score_chunk', fake)


def _metrics(app, clip_id):
    from app.models import VideoClip
    with app.app_context():
        raw = VideoClip.query.get(clip_id).metrics_json
    return json.loads(raw) if raw else {}


def test_a_measured_shot_stores_the_score_the_state_and_the_frame_count(app, monkeypatch):
    bank_id, ids = _bank_with_clips(app, _long(1))
    _stub_decode(monkeypatch)
    _stub_model(monkeypatch, lambda cid: [1.0, 2.0, 3.0] + [2.0] * 12)
    with app.app_context():
        out = ac.run_ai_check(bank_id)
    assert out == {'measured': 1, 'too_short': 0, 'unreadable': 0, 'error': None}
    stored = _metrics(app, ids[0])
    assert stored[ac.STATE_KEY] == 'ok'
    # The frame count travels with the score because the score's SCALE depends on
    # it: a reader comparing two banks has to be able to see they were measured
    # the same way.
    assert stored[ac.FRAMES_KEY] == ac.FRAMES
    assert stored[ac.SCORE_KEY] == pytest.approx(
        round(ac.irregularity([1.0, 2.0, 3.0] + [2.0] * 12), 4))


def test_a_shot_too_short_for_the_window_is_a_state_and_carries_no_score(app, monkeypatch):
    """Never a zero, and never re-decoded on the next run: the cut is what is too
    short, and re-running will not change it. Trimming it would."""
    bank_id, ids = _bank_with_clips(app, [(0.0, 1.0)])
    _stub_decode(monkeypatch)
    _stub_model(monkeypatch, lambda cid: [2.0] * 15)
    with app.app_context():
        out = ac.run_ai_check(bank_id)
    assert out['too_short'] == 1 and out['measured'] == 0
    stored = _metrics(app, ids[0])
    assert stored[ac.STATE_KEY] == 'too_short'
    assert ac.SCORE_KEY not in stored
    # And it stays out of the queue rather than costing a decode every run.
    with app.app_context():
        assert ac.pending_clips(bank_id) == []


def test_a_window_that_will_not_decode_is_a_state_and_carries_no_score(app, monkeypatch):
    bank_id, ids = _bank_with_clips(app, _long(1))
    def boom(path, times, dest, stem):
        raise RuntimeError('only 3 of 16 frames decoded')
    monkeypatch.setattr(ac, '_write_window', boom)
    _stub_model(monkeypatch, lambda cid: [2.0] * 15)
    with app.app_context():
        out = ac.run_ai_check(bank_id)
    assert out['unreadable'] == 1
    stored = _metrics(app, ids[0])
    assert stored[ac.STATE_KEY] == 'unreadable'
    assert ac.SCORE_KEY not in stored


def test_a_shot_the_child_skipped_keeps_no_key_and_goes_back_in_the_queue(app, monkeypatch):
    """The difference between a STATE and an ABSENCE, and it is the resume
    contract. 'too_short' and 'unreadable' are decisions about the shot and are
    written down. A shot the model never answered for is not a decision — it is
    unfinished work, and the only thing that puts it back in the next run's queue
    is having no key at all."""
    bank_id, ids = _bank_with_clips(app, _long(2))
    _stub_decode(monkeypatch)
    _stub_model(monkeypatch,
                lambda cid: [2.0, 3.0] + [1.0] * 13 if cid == ids[0] else None)
    with app.app_context():
        out = ac.run_ai_check(bank_id)
        assert out['measured'] == 1
        assert _metrics(app, ids[1]) == {}
        assert [c.id for c in ac.pending_clips(bank_id)] == [ids[1]]


def test_a_partial_window_is_no_measurement_rather_than_a_smaller_one(app, monkeypatch):
    """A standard deviation over six samples and one over fourteen are not on the
    same scale, so a single cut across them would flag shots for their length.
    A short series is refused even though `irregularity` could compute one."""
    bank_id, ids = _bank_with_clips(app, _long(1))
    _stub_decode(monkeypatch)
    _stub_model(monkeypatch, lambda cid: [2.0, 3.0, 1.0, 4.0, 2.0])
    with app.app_context():
        out = ac.run_ai_check(bank_id)
    assert out['measured'] == 0
    assert _metrics(app, ids[0]) == {}


def test_the_pass_merges_into_the_blob_and_never_erases_another_passs_verdict(app, monkeypatch):
    from app.extensions import db
    from app.models import VideoClip
    bank_id, ids = _bank_with_clips(app, _long(1))
    with app.app_context():
        clip = VideoClip.query.get(ids[0])
        clip.metrics_json = json.dumps({'sharpness_p90': 312.5,
                                        'watermark_score': 0.98,
                                        ac.SCORE_KEY: 99.0,
                                        ac.STATE_KEY: 'unreadable'})
        db.session.commit()
    _stub_decode(monkeypatch)
    _stub_model(monkeypatch, lambda cid: [1.0, 3.0, 2.0] + [2.0] * 12)
    with app.app_context():
        ac.run_ai_check(bank_id, rescan=True)
    stored = _metrics(app, ids[0])
    assert stored['sharpness_p90'] == 312.5
    assert stored['watermark_score'] == 0.98
    assert stored[ac.STATE_KEY] == 'ok'
    assert stored[ac.SCORE_KEY] != 99.0


def test_a_recheck_that_now_finds_a_shot_too_short_leaves_no_stale_score(app, monkeypatch):
    """The keys this pass owns are cleared before the new ones are written, so a
    'too_short' state can never sit next to last run's number."""
    from app.extensions import db
    from app.models import VideoClip
    bank_id, ids = _bank_with_clips(app, [(0.0, 1.0)])
    with app.app_context():
        clip = VideoClip.query.get(ids[0])
        clip.metrics_json = json.dumps({ac.STATE_KEY: 'ok', ac.SCORE_KEY: 1.23,
                                        ac.FRAMES_KEY: 16})
        db.session.commit()
        ac.run_ai_check(bank_id, rescan=True)
    stored = _metrics(app, ids[0])
    assert stored == {ac.STATE_KEY: 'too_short'}


def test_the_model_failing_is_a_result_that_keeps_what_it_already_measured(app, monkeypatch):
    """A machine with no egress must not have its whole run reported as failed
    because a first-run download did not come back — and the sentence has to
    survive to the UI, because '0 flagged' on its own reads as a clean bank."""
    bank_id, ids = _bank_with_clips(app, _long(ac.CHUNK + 1))
    _stub_decode(monkeypatch)
    calls = {'n': 0}

    def flaky(payload, *, timeout=None):
        calls['n'] += 1
        if calls['n'] == 1:
            return {item['id']: [2.0, 4.0] + [3.0] * 13 for item in payload}
        raise RuntimeError('could not load microsoft/xclip-base-patch16: offline')

    monkeypatch.setattr(ac, 'score_chunk', flaky)
    with app.app_context():
        out = ac.run_ai_check(bank_id)
    assert out['measured'] == ac.CHUNK
    assert 'offline' in out['error']
    assert _metrics(app, ids[0])[ac.STATE_KEY] == 'ok'
    assert _metrics(app, ids[-1]) == {}


def test_stopping_keeps_every_shot_already_measured(app, monkeypatch):
    bank_id, ids = _bank_with_clips(app, _long(ac.CHUNK + 1))
    _stub_decode(monkeypatch)
    _stub_model(monkeypatch, lambda cid: [1.0, 2.5] + [2.0] * 13)
    stop = {'now': False}
    with app.app_context():
        out = ac.run_ai_check(bank_id, should_stop=lambda: stop['now'],
                              on_clip=lambda: stop.__setitem__('now', True))
    assert out['measured'] >= 1
    assert _metrics(app, ids[0])[ac.STATE_KEY] == 'ok'


def test_only_shots_of_a_file_that_probed_are_queued(app, monkeypatch):
    """An unreadable file has no frame to decode, and counting its shots as
    unreadable on every run would make the pass look permanently broken."""
    bank_id, _ids = _bank_with_clips(app, _long(3), probe_state='unreadable')
    with app.app_context():
        assert ac.pending_clips(bank_id) == []


# --- the verdict, at read time --------------------------------------------------------

def test_a_low_score_is_the_suspicious_one_and_the_cut_is_a_floor():
    """The polarity, on the rule the grid actually runs. Inverting this would
    flag every handheld shot in a bank and clear every generated one."""
    smooth = {ac.SCORE_KEY: 0.4}
    erratic = {ac.SCORE_KEY: 2.1}
    cut = {'motion_irregularity_floor': 1.0}
    assert 'maybe_generated' in video_metrics.verdicts(smooth, cut)
    assert 'maybe_generated' not in video_metrics.verdicts(erratic, cut)


def test_moving_the_cut_re_sorts_the_bank_with_nothing_rescanned():
    scores = {ac.SCORE_KEY: 1.5}
    assert 'maybe_generated' not in video_metrics.verdicts(scores, {'motion_irregularity_floor': 1.0})
    assert 'maybe_generated' in video_metrics.verdicts(scores, {'motion_irregularity_floor': 2.0})


def test_an_unchecked_shot_is_never_flagged():
    """No measurement, no verdict — the rule every cut in this module obeys. A
    bank checked before this shipped, and a shot too short for the window, must
    not be swept up by somebody setting a cut."""
    for scores in ({}, {ac.STATE_KEY: 'too_short'}, {ac.STATE_KEY: 'unreadable'},
                   {'sharpness_p90': 100.0}):
        assert video_metrics.verdicts(scores, {'motion_irregularity_floor': 5.0}) == set()


def test_no_cut_flags_nothing_however_smooth_the_shot():
    assert video_metrics.verdicts({ac.SCORE_KEY: 0.0}, {}) == set()
    assert video_metrics.verdicts({ac.SCORE_KEY: 0.0},
                                  {'motion_irregularity_floor': None}) == set()


def test_the_cut_ships_empty_and_is_the_panels_to_offer():
    from app import config as cfg
    assert cfg.DEFAULTS['video_bank']['motion_irregularity_floor'] is None
    assert 'motion_irregularity_floor' in video_metrics.THRESHOLD_KEYS


def test_the_dry_run_counts_this_cut_like_every_other():
    counts = video_metrics.dry_run(
        [{ac.SCORE_KEY: 0.2}, {ac.SCORE_KEY: 3.0}, {}],
        {'motion_irregularity_floor': 1.0})
    assert counts['maybe_generated'] == 1
    assert counts['total_flagged'] == 1


# --- the keys, pinned against the passes that share the blob --------------------------

def test_every_key_this_pass_owns_survives_a_re_measure():
    """A key this pass writes and ADVISORY_KEYS does not carry is erased by the
    next quality scan, silently — and re-earning it costs tens of minutes."""
    missing = [k for k in ac.OWNED_KEYS if k not in video_metrics.ADVISORY_KEYS]
    assert not missing, (f'{missing} would be erased by the metrics pass — add '
                         f'them to video_metrics.ADVISORY_KEYS')


def test_the_state_key_is_owned_so_a_recheck_cannot_leave_two_answers():
    assert ac.STATE_KEY in ac.OWNED_KEYS
    assert ac.SCORE_KEY in ac.OWNED_KEYS


def test_this_pass_does_not_write_a_key_another_pass_owns():
    """The blob is shared by seven passes now. An overlap would make one pass's
    re-run silently rewrite another's verdict."""
    from app.services import video_defect_sweep, video_safe_zone
    others = set(video_safe_zone.OWNED_KEYS) | set(video_defect_sweep.OWNED_KEYS)
    others |= {'aesthetic_score', 'watermark_score', 'watermark_state',
               'duplicate_group', 'duplicate_of', video_metrics.SHARPNESS_KEY}
    assert not (set(ac.OWNED_KEYS) & others)


# --- the port, pinned so it cannot drift by accident ----------------------------------

def test_the_worker_reproduces_the_references_preprocessing_including_its_quirks(tmp_path):
    """The two deliberate reproductions, checked by RUNNING the preprocessing
    rather than by grepping for it — the source names the alternatives it does
    not use, so a text search finds them and calls the port broken.

    A flat RGB(200, 100, 50) frame. The reference reads with `cv2.imread`, which
    is BGR, so the first channel must come back holding the BLUE 50 and not the
    red 200; and it normalises with ImageNet statistics rather than CLIP's own,
    so the exact values are decided too. Both look like bugs and both are the
    only configuration measured against real generators — "fixing" one takes the
    feature off its evidence, which is why this is pinned."""
    pytest.importorskip('numpy')
    Image = pytest.importorskip('PIL.Image')
    infer = _infer_module()

    path = str(tmp_path / 'flat.jpg')
    Image.new('RGB', (16, 16), (200, 100, 50)).save(path, 'JPEG', quality=95)
    out = infer._preprocess([path])
    assert out.shape == (1, 3, 16, 16)          # one frame, CHW
    channels = [float(out[0, c].mean()) for c in range(3)]

    def expect(value, mean, std):
        return (value / 255.0 - mean) / std

    assert channels[0] == pytest.approx(expect(50, 0.485, 0.229), abs=0.05),         'the first channel is not BLUE — the reference reads BGR and this does not'
    assert channels[1] == pytest.approx(expect(100, 0.456, 0.224), abs=0.05)
    assert channels[2] == pytest.approx(expect(200, 0.406, 0.225), abs=0.05)
    # And decisively NOT the CLIP normalisation the encoder was trained with.
    assert channels[0] != pytest.approx(expect(50, 0.4815, 0.2686), abs=0.02)


def test_the_worker_takes_the_unprojected_unnormalised_pooled_feature():
    """`pooler_output` is post_layernorm(CLS): 768-d, NOT projected into the joint
    text space and NOT L2-normalised.

    The behavioural half is the one that matters: normalising the features would
    put every vector on the unit sphere and flatten exactly the magnitude this
    statistic reads. So scaling the features must scale the distances — the look
    score's child re-norms deliberately, and copying that reflex here would
    quietly destroy the signal."""
    pytest.importorskip('numpy')
    infer = _infer_module()
    features = [[0.0, 0.0], [1.0, 0.0], [4.0, 0.0], [4.0, 4.0]]
    doubled = [[2 * v for v in row] for row in features]
    assert infer.step_distances(doubled) == pytest.approx(
        [2 * d for d in infer.step_distances(features)])
    # A normalising implementation would answer the same thing for both.
    assert infer.step_distances(doubled) != pytest.approx(
        infer.step_distances(features))
    assert 'pooler_output' in _infer_code('video_ai_check_infer.py')


def test_the_worker_hides_the_card_before_it_imports_torch():
    """Order matters: setting the variable after the import does nothing. Same
    two-locks-on-one-door reflex as the look score's child."""
    src = _infer_source('video_ai_check_infer.py')
    assert (src.index("os.environ['CUDA_VISIBLE_DEVICES'] = ''")
            < src.index('import torch'))


def test_the_worker_takes_the_tower_off_the_whole_model_never_loads_it_alone():
    """The bug that would have shipped a feature measuring noise.

    The published checkpoint stores its vision tensors under a `vision_model.`
    prefix. `XCLIPVisionModel.from_pretrained(id)` — the obvious call, and the
    reference's own — therefore matches NOTHING and returns a randomly
    initialised encoder. Measured on transformers 5.14.1 against the raw
    safetensors: 343 missing keys that way, 0 through `XCLIPModel`.

    It fails silently in every way that matters: no raise, plausible weight
    statistics (patch-embedding std 0.0200 against the checkpoint's 0.0189), and
    two forged clips still separated 0.09 against 0.28 — a random projection
    keeps enough of the structure to look like a working detector. So the
    loading class is pinned here rather than left to read correctly.

    Read with `ast` rather than by substring, because the docstring above has to
    NAME the wrong call in order to warn about it — a text search would forbid
    the warning along with the bug."""
    import ast

    src = _infer_source('video_ai_check_infer.py')
    loaders = {node.func.value.id
               for node in ast.walk(ast.parse(src))
               if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Attribute)
               and node.func.attr == 'from_pretrained'
               and isinstance(node.func.value, ast.Name)}
    assert loaders == {'XCLIPModel'}, \
        f'the tower must come off the whole model; loaders found: {loaders}'
    assert '.vision_model' in src


def test_the_worker_refuses_an_encoder_whose_weights_did_not_arrive():
    """The guard behind the test above, and it is version-portable in a way that
    pinning a class name is not: whatever transformers does with prefixes next,
    `missing_keys` says whether the weights arrived, and a non-empty one is
    REFUSED rather than reported. Storing noise costs the user's trust in every
    score in the bank; stopping with a reason costs one run."""
    src = _infer_source('video_ai_check_infer.py')
    assert 'output_loading_info=True' in src
    assert "info.get('missing_keys')" in src
    body = src.split('def main')[1]
    guard = body.split("missing = info.get('missing_keys')")[1][:600]
    assert 'raise' in guard, 'a missing weight must stop the run, not be logged'
    assert 'randomly initialised' in guard, \
        'the error has to say WHY the numbers would be worthless'


def test_batching_never_straddles_two_clips():
    """The cross-frame message block groups by `num_frames` consecutive batch
    items. Several clips may share a forward pass only while each contributes a
    multiple of that — otherwise one shot's motion leaks into another's, and the
    numbers would still look perfectly plausible."""
    pytest.importorskip('numpy')
    import numpy as np
    infer = _infer_module()

    class _Cfg:
        num_frames = 8

    class _Model:
        config = _Cfg()

        def __call__(self, **kwargs):
            raise AssertionError('the guard should have refused before this')

    with pytest.raises(ValueError, match='multiple of 8'):
        infer._encode(_Model(), np.zeros((12, 3, 4, 4), dtype='float32'), [12])
    # And 16 is accepted, which is what the parent always sends.
    class _Ok(_Model):
        def __call__(self, **kwargs):
            raise RuntimeError('reached the model')
    with pytest.raises(RuntimeError, match='reached the model'):
        infer._encode(_Ok(), np.zeros((32, 3, 4, 4), dtype='float32'), [16, 16])


def test_the_encoder_is_a_constant_and_not_a_setting():
    """Unlike `video_caption.model`. Swapping the encoder moves the scale under
    every number already stored and under any cut set against it, so it is a
    migration rather than a preference — and a setting would invite it."""
    from app import config as cfg
    assert ac.MODEL_ID == 'microsoft/xclip-base-patch16'
    assert 'ai_check' not in cfg.DEFAULTS
    assert 'model' not in str(cfg.DEFAULTS.get('bank_scoring', {}))


def test_the_probe_of_the_interpreter_this_runs_in_imports_what_the_worker_imports():
    """CLAUDE.md's rule, and issue #24's shape. This pass borrows the ✨ Score
    interpreter rather than declaring one of its own, so the honest check is that
    THAT probe covers these imports — a probe that names only the headline
    packages reports ✓ while the feature dies on the first call."""
    from app.capabilities import CAPABILITY_IMPORTS
    probe = CAPABILITY_IMPORTS['bank_scoring']
    for name in ('torch', 'transformers', 'numpy'):
        assert name in probe, f'the worker imports {name} and the probe does not'
    assert 'from PIL import Image' in probe, \
        'the bare package imports without its submodules and would pass broken'


def test_the_install_verifies_with_the_probes_own_import_not_a_shorter_one(monkeypatch):
    """The other half of CLAUDE.md's rule: *never let an install claim success
    without re-running that probe*. Widening the probe above is only half a fix
    if ✨ Score's install gate keeps checking a shorter list — it would print
    "ready" and be contradicted seconds later by a ✗ with no reason anywhere,
    which is issue #24 wearing the install's clothes rather than the probe's.
    So the gate runs the probe STRING, and this pins that rather than the words
    in it: a name added to the probe tomorrow is checked by the install with no
    second edit."""
    from app import capabilities, setup_installer

    ran = {}

    def fake_run(argv, **kwargs):
        ran['argv'] = list(argv)
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    monkeypatch.setattr(setup_installer.subprocess, 'run', fake_run)
    monkeypatch.setattr(setup_installer.os.path, 'isfile', lambda p: True)
    monkeypatch.setattr(setup_installer, '_append', lambda *a, **k: None)
    assert setup_installer._verify_bank_scoring_import('bank_scoring', 'py.exe')
    assert ran['argv'][-1] == capabilities.CAPABILITY_IMPORTS['bank_scoring']
    # `-s` survives: this interpreter is PROBED with no user site
    # (capabilities._NO_USER_SITE_IMPORT_KEYS), and a verification run with
    # different argv answers a different question than the one that decides ✓/✗.
    assert '-s' in ran['argv']


def test_the_pass_needs_no_install_action_of_its_own():
    """It adds no pip package: `transformers` is already what the ✨ Score
    extra installs, and the encoder's weights are a first-run download the job
    line warns about — the same shape as the caption model. Nothing new for
    Setup means nothing new to forget to wire."""
    from app import setup_installer
    assert 'transformers' in setup_installer._BANK_SCORING_PKGS
    assert 'ai_check' not in setup_installer.INSTALL_ACTIONS
    assert 'aicheck' not in setup_installer.INSTALL_ACTIONS


# --- the route and the job ------------------------------------------------------------

def test_the_route_starts_the_pass_and_the_button_has_a_name(app, client, monkeypatch):
    bank_id, _ids = _bank_with_clips(app, _long(1))
    started = {}
    monkeypatch.setattr('app.services.video_bank_service.start_ai_check',
                        lambda *a, **k: started.update(k) or {'ok': True})
    monkeypatch.setattr('app.capabilities.probe_video',
                        lambda: {'decode': True, 'detect': True, 'encode': True,
                                 'detail': ''})
    r = client.post(f'/api/video-bank/{bank_id}/aicheck', json={'recheck': True})
    assert r.status_code in (200, 202), r.data
    assert started.get('recheck') is True


def test_the_route_refuses_without_the_decode_extra(app, client, monkeypatch):
    bank_id, _ids = _bank_with_clips(app, _long(1))
    monkeypatch.setattr('app.capabilities.probe_video',
                        lambda: {'decode': False, 'detect': False, 'encode': False,
                                 'detail': 'install the video extra from Setup'})
    r = client.post(f'/api/video-bank/{bank_id}/aicheck', json={})
    assert r.status_code == 503
    assert 'video extra' in r.get_json()['error']


def test_the_pass_refuses_up_front_when_no_interpreter_can_run_the_encoder(app, monkeypatch):
    """A 202 followed by a job that dies on an import is the same news, twenty
    minutes later and harder to read."""
    bank_id, _ids = _bank_with_clips(app, _long(1))
    monkeypatch.setattr(ac, 'unavailable_reason',
                        lambda: 'the AI check needs the ✨ Score interpreter')
    from app.services import video_bank_service as svc
    with app.app_context():
        with pytest.raises(RuntimeError, match='Score interpreter'):
            svc.start_ai_check(app, cfg.LOCAL_USER, bank_id)


def test_the_pass_takes_no_gpu_window(app):
    """It runs for tens of minutes; holding the card that long over an advisory
    flag would unload ComfyUI and block a training start for the whole run. Not
    touching it is what lets a bank be checked WHILE a training owns the card."""
    src = (Path(__file__).resolve().parents[1] / 'app' / 'services'
           / 'video_bank_service.py').read_text(encoding='utf-8')
    body = src.split('def start_ai_check')[1].split('def _caption_available')[0]
    assert 'gpu_exclusive' not in body
    assert '_gpu_busy_reason' not in body


# --- the real decode, when the extra is present ---------------------------------------

def test_the_real_decode_writes_one_square_frame_per_instant(tmp_path):
    """Integration: the true PyAV path on a synthesised clip. Skips — loudly —
    when the video extra is absent. What it proves is the part a stub cannot:
    that a contiguous 8 fps window really comes back as FRAMES files at the
    encoder's own input size."""
    av = pytest.importorskip('av', reason='video extra not installed')
    Image = pytest.importorskip('PIL.Image', reason='Pillow not installed')
    import numpy as np

    path = str(tmp_path / 'clip.mp4')
    with av.open(path, 'w') as container:
        stream = container.add_stream('h264', rate=25)
        stream.width, stream.height, stream.pix_fmt = 320, 180, 'yuv420p'
        for i in range(150):                      # 6 seconds
            img = np.full((180, 320, 3), (i * 3) % 255, dtype=np.uint8)
            img[:, (i * 2) % 300:(i * 2) % 300 + 20] = 255
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format='rgb24')):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    times = ac.window_times(0.0, 6.0)
    paths = ac._write_window(path, times, str(tmp_path / 'frames'), 'clip_1')
    assert len(paths) == ac.FRAMES
    with Image.open(paths[0]) as im:
        assert im.size == (ac.FRAME_SIDE, ac.FRAME_SIDE)


@pytest.mark.live_xclip
def test_the_real_encoder_separates_smooth_motion_from_erratic_motion(tmp_path):
    """The end-to-end proof, opt-in because it downloads several hundred
    megabytes of weights: LDS_TEST_LIVE_XCLIP=1.

    Two forged windows of the SAME content — one panned at a constant speed, one
    panned with per-frame jitter — through the real child. The smooth one must
    score lower. This is the assertion the whole feature rests on, and it is the
    one no stub can make.

    `ok` implies REAL WEIGHTS, which is why it is asserted first and with the
    child's own error: the loader refuses an encoder with missing keys. That
    matters more than it reads, because polarity alone proves nothing here — a
    randomly initialised encoder passed this same comparison 0.09 against 0.28.
    Ordering is a weak signal; the weight guard is the strong one."""
    import os
    if os.environ.get('LDS_TEST_LIVE_XCLIP') != '1':
        pytest.skip('set LDS_TEST_LIVE_XCLIP=1 to download the encoder and run')
    Image = pytest.importorskip('PIL.Image')
    import random
    import subprocess

    def window(jitter, seed):
        rng = random.Random(seed)
        out = []
        base = Image.new('RGB', (900, 500))
        pixels = base.load()
        for y in range(500):
            for x in range(900):
                pixels[x, y] = ((x * 7 + y * 13) % 256, (x * 3) % 256, (y * 5) % 256)
        folder = tmp_path / f'w{jitter}'
        folder.mkdir()
        for i in range(ac.FRAMES):
            x = 20 + 12 * i + (rng.randint(-6, 6) if jitter else 0)
            frame = base.crop((x, 40, x + 400, 40 + 300))
            frame = frame.resize((ac.FRAME_SIDE, ac.FRAME_SIDE), Image.BILINEAR)
            path = str(folder / f'{i:02d}.jpg')
            frame.save(path, 'JPEG', quality=95)
            out.append(path)
        return out

    payload = json.dumps({'clips': [{'id': 1, 'frames': window(False, 1)},
                                    {'id': 2, 'frames': window(True, 2)}],
                          'model': ac.MODEL_ID, 'models_root': None})
    # The ✨ Score interpreter, NOT sys.executable. The suite's own Python is the
    # Flask one and has no torch, so pointing this at it makes the opt-in test
    # unrunnable on the machine most likely to be asked to run it.
    python = cfg.get('bank_scoring.python') or sys.executable
    proc = subprocess.run([python, str(_infer_dir() / 'video_ai_check_infer.py')],
                          input=payload + '\n', capture_output=True, text=True,
                          encoding='utf-8', errors='replace')
    lines = [line for line in (proc.stdout or '').strip().splitlines()
             if line.lstrip().startswith('{')]
    assert lines, f'no JSON from the worker: {(proc.stderr or "")[-600:]}'
    data = json.loads(lines[-1])
    # `ok` false here is usually the weight guard, and its message is the whole
    # point of the run — a bare assert would hide the one sentence worth reading.
    assert data['ok'], data.get('error')
    smooth = ac.irregularity(data['steps']['1'])
    erratic = ac.irregularity(data['steps']['2'])
    assert smooth < erratic, (smooth, erratic)
