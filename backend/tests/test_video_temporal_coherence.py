"""🔗 Does one shot hold ONE scene — the arithmetic, the states, and the cut.

Every vector here is FORGED in arithmetic, so the true answer is known exactly
and no encoder, no .npz and no footage sits between the intent and the assertion
— the same split the camera pass draws around `clip_steps`.
"""
import json

import pytest

from app.extensions import db
from app.models import VideoBank, VideoClip, VideoSource
from app.services import video_metrics, video_temporal_coherence as coherence


# --- forged vectors -----------------------------------------------------------------

def vec(*components, dim=8):
    """A unit vector in a small space. The arithmetic is dimension-agnostic and
    768 floats per fixture would hide the numbers this file is about."""
    import numpy as np
    v = np.zeros(dim, dtype='float32')
    for i, c in enumerate(components):
        v[i] = c
    return v / (np.linalg.norm(v) + 1e-12)


def frames(*pairs):
    """[{label, time_s, vec}] from (time, vector) pairs, deliberately out of time
    order in some tests — the pass sorts, and that is a contract."""
    return [{'label': f'f{i}', 'time_s': t, 'vec': v}
            for i, (t, v) in enumerate(pairs)]


def test_one_scene_throughout_reads_near_one():
    """A shot whose three frames are the same picture is perfectly coherent."""
    same = vec(1, 0)
    out = coherence.coherence_of(frames((0.25, same), (1.0, same), (1.75, same)))
    assert out['coherence_state'] == 'ok'
    assert out['coherence_span'] == pytest.approx(1.0, abs=1e-4)
    assert out['coherence_min'] == pytest.approx(1.0, abs=1e-4)
    assert out['coherence_frames'] == 3


def test_two_scenes_read_as_divergence_not_as_stillness():
    """THE POLARITY, pinned. Two orthogonal halves is the maximum divergence two
    unit vectors can show short of pointing away from each other, and it must
    come out LOW — the roadmap note that specified this pass had the two cases
    the other way round."""
    out = coherence.coherence_of(frames(
        (0.25, vec(1, 0)), (1.0, vec(1, 0)), (1.75, vec(0, 1))))
    assert out['coherence_span'] == pytest.approx(0.0, abs=1e-4)
    assert out['coherence_span'] < out['coherence_min'] + 1e-9


def test_an_ordinary_shot_lands_between_the_two():
    """Drift without a cut: the span is high but not 1. This is the case a floor
    has to leave alone, and it is the bulk of any real bank."""
    out = coherence.coherence_of(frames(
        (0.25, vec(1, 0.0)), (1.0, vec(1, 0.2)), (1.75, vec(1, 0.45))))
    assert 0.85 < out['coherence_span'] < 0.99


def test_the_verdict_is_the_first_to_last_pair_and_not_the_minimum():
    """The measured choice, pinned so it cannot be 'simplified' into the min.

    A shot that wanders away at its MIDDLE frame and comes back has a low
    minimum and a high span — and it is one scene. Storing the min as the verdict
    would flag it."""
    out = coherence.coherence_of(frames(
        (0.25, vec(1, 0)), (1.0, vec(0, 1)), (1.75, vec(1, 0))))
    assert out['coherence_span'] == pytest.approx(1.0, abs=1e-4)
    assert out['coherence_min'] == pytest.approx(0.0, abs=1e-4)
    # And the flag reads the span, so this shot is NOT flagged at a cut that the
    # minimum would trip twice over.
    assert 'missed_cut' not in video_metrics.verdicts(out, {'coherence_floor': 0.8})


def test_frames_are_compared_in_time_order_not_in_store_order():
    """The store's order is whatever the embed pass wrote. First-to-last means
    first and last IN TIME, or the span is a random pair."""
    a, b = vec(1, 0), vec(0, 1)
    shuffled = coherence.coherence_of(frames((1.75, b), (0.25, a), (1.0, a)))
    ordered = coherence.coherence_of(frames((0.25, a), (1.0, a), (1.75, b)))
    assert shuffled['coherence_span'] == ordered['coherence_span']
    assert shuffled['coherence_span_s'] == pytest.approx(1.5)


def test_the_compared_span_is_stored_in_seconds():
    """The confound the calibration found — this cosine falls with elapsed time —
    is not recoverable from the clip bounds, because the embedded frames sit
    inside them. So the seconds are stored."""
    out = coherence.coherence_of(frames(
        (0.25, vec(1, 0)), (2.0, vec(1, 0)), (9.75, vec(1, 0))))
    assert out['coherence_span_s'] == pytest.approx(9.5)


def test_unnormalised_vectors_cannot_manufacture_a_similarity_above_one():
    """Normed here rather than trusted from the store, like the dedup pass."""
    import numpy as np
    big = np.array([3.0, 0, 0, 0, 0, 0, 0, 0], dtype='float32')
    out = coherence.coherence_of(frames((0.0, big), (1.0, big)))
    assert out['coherence_span'] == pytest.approx(1.0, abs=1e-4)


# --- degraded inputs ------------------------------------------------------------------

def test_a_shot_with_one_frame_gets_a_state_and_no_numbers():
    """Under MIN_SPAN_FOR_THREE_S the embed pass writes ONE frame. There is no
    second instant, so there is no number — and a 1.0 would be the app asserting
    perfect coherence about a span nobody sampled twice."""
    out = coherence.coherence_of(frames((0.5, vec(1, 0))))
    assert out == {'coherence_state': 'one_frame'}
    assert 'missed_cut' not in video_metrics.verdicts(out, {'coherence_floor': 0.9})


def test_two_frames_are_enough_and_are_reported_as_two():
    """A store can hold two — one of three frames failed to decode. That is a
    measurement, not a failure, and the count says which it was."""
    out = coherence.coherence_of(frames((0.25, vec(1, 0)), (1.75, vec(0, 1))))
    assert out['coherence_state'] == 'ok'
    assert out['coherence_frames'] == 2
    assert out['coherence_span'] == pytest.approx(0.0, abs=1e-4)


def test_no_vectors_at_all_is_none_rather_than_a_state():
    """The two absences are different and must not collapse: 'one_frame' retires
    a shot from the queue, None re-queues it for the run that repairs the
    store."""
    assert coherence.coherence_of([]) is None
    assert coherence.coherence_of(None) is None


# --- the pass over a bank --------------------------------------------------------------

@pytest.fixture()
def bank(app):
    with app.app_context():
        b = VideoBank(user_id=1, name='b', source_path='x')
        db.session.add(b)
        db.session.flush()
        s = VideoSource(bank_id=b.id, relpath='a.mp4', probe_state='ok')
        db.session.add(s)
        db.session.flush()
        for i in range(4):
            db.session.add(VideoClip(bank_id=b.id, source_id=s.id,
                                     start_s=i * 3.0, end_s=i * 3.0 + 2.5,
                                     embed_state='ok'))
        db.session.commit()
        yield b.id


def clips_of(bank_id):
    return VideoClip.query.filter_by(bank_id=bank_id).order_by(VideoClip.id).all()


def test_the_pass_reads_the_store_and_reports_what_it_could_not(app, bank, monkeypatch):
    with app.app_context():
        rows = clips_of(bank)
        a, b, c, d = rows
        store = {
            a.id: frames((0.25, vec(1, 0)), (1.0, vec(1, 0)), (2.25, vec(1, 0))),
            b.id: frames((0.25, vec(1, 0)), (1.0, vec(1, 0)), (2.25, vec(0, 1))),
            c.id: frames((0.5, vec(1, 0))),
            # d has no vectors: the .npz and the database disagree.
        }
        monkeypatch.setattr(coherence, 'load_vectors', lambda _b: store)
        out = coherence.run_coherence(bank)
        assert out == {'measured': 2, 'one_frame': 1, 'unmeasured': 1}

        got = {r.id: json.loads(r.metrics_json or '{}') for r in clips_of(bank)}
        assert got[a.id]['coherence_span'] == pytest.approx(1.0, abs=1e-4)
        assert got[b.id]['coherence_span'] == pytest.approx(0.0, abs=1e-4)
        assert got[c.id]['coherence_state'] == 'one_frame'
        # NO KEY, not a zero and not a state: that absence is the resume test.
        assert 'coherence_state' not in got[d.id]


def test_a_shot_with_no_vectors_comes_back_next_run(app, bank, monkeypatch):
    """The desynchronised store repairs itself once the vectors arrive."""
    with app.app_context():
        rows = clips_of(bank)
        monkeypatch.setattr(coherence, 'load_vectors', lambda _b: {})
        assert coherence.run_coherence(bank)['unmeasured'] == 4
        assert len(coherence.pending_clips(bank)) == 4

        store = {r.id: frames((0.25, vec(1, 0)), (2.25, vec(1, 0))) for r in rows}
        monkeypatch.setattr(coherence, 'load_vectors', lambda _b: store)
        assert coherence.run_coherence(bank)['measured'] == 4
        assert coherence.pending_clips(bank) == []


def test_only_shots_that_still_claim_their_vectors_are_read(app, bank, monkeypatch):
    """`embed_state` is the authority. A re-cut shot has vectors on disk for a
    span it no longer has, and judging it on them is judging deleted footage."""
    with app.app_context():
        rows = clips_of(bank)
        rows[0].embed_state = None
        db.session.commit()
        store = {r.id: frames((0.25, vec(1, 0)), (2.25, vec(0, 1))) for r in rows}
        monkeypatch.setattr(coherence, 'load_vectors', lambda _b: store)
        out = coherence.run_coherence(bank)
        assert out['measured'] == 3
        assert json.loads(clips_of(bank)[0].metrics_json or '{}') == {}


def test_a_rerun_only_pays_for_what_the_first_run_did_not_reach(app, bank, monkeypatch):
    with app.app_context():
        rows = clips_of(bank)
        store = {r.id: frames((0.25, vec(1, 0)), (2.25, vec(1, 0))) for r in rows}
        monkeypatch.setattr(coherence, 'load_vectors', lambda _b: store)
        stop = {'n': 0}

        def should_stop():
            stop['n'] += 1
            return stop['n'] > 2
        coherence.run_coherence(bank, should_stop=should_stop)
        assert len(coherence.pending_clips(bank)) == 2
        assert coherence.run_coherence(bank)['measured'] == 2


def test_recheck_reads_every_shot_again(app, bank, monkeypatch):
    """Rewritten vectors are different vectors — a stale reading beside them is a
    verdict about footage that has moved."""
    with app.app_context():
        rows = clips_of(bank)
        store = {r.id: frames((0.25, vec(1, 0)), (2.25, vec(1, 0))) for r in rows}
        monkeypatch.setattr(coherence, 'load_vectors', lambda _b: store)
        coherence.run_coherence(bank)
        assert coherence.pending_clips(bank) == []
        assert len(coherence.pending_clips(bank, recheck=True)) == 4

        store = {r.id: frames((0.25, vec(1, 0)), (2.25, vec(0, 1))) for r in rows}
        coherence.run_coherence(bank, recheck=True)
        got = json.loads(clips_of(bank)[0].metrics_json)
        assert got['coherence_span'] == pytest.approx(0.0, abs=1e-4)


def test_a_rerun_that_now_finds_one_frame_leaves_no_stale_numbers(app, bank, monkeypatch):
    with app.app_context():
        rows = clips_of(bank)
        store = {rows[0].id: frames((0.25, vec(1, 0)), (2.25, vec(0, 1)))}
        monkeypatch.setattr(coherence, 'load_vectors', lambda _b: store)
        coherence.run_coherence(bank)
        store[rows[0].id] = frames((0.5, vec(1, 0)))
        coherence.run_coherence(bank, recheck=True)
        got = json.loads(clips_of(bank)[0].metrics_json)
        assert got['coherence_state'] == 'one_frame'
        assert 'coherence_span' not in got
        assert 'coherence_min' not in got


def test_the_pass_merges_and_never_erases_another_pass(app, bank, monkeypatch):
    with app.app_context():
        rows = clips_of(bank)
        rows[0].metrics_json = json.dumps({'metrics_state': 'ok',
                                           'sharpness_p90': 120.0,
                                           'aesthetic_score': 5.5})
        db.session.commit()
        store = {rows[0].id: frames((0.25, vec(1, 0)), (2.25, vec(1, 0)))}
        monkeypatch.setattr(coherence, 'load_vectors', lambda _b: store)
        coherence.run_coherence(bank)
        got = json.loads(clips_of(bank)[0].metrics_json)
        assert got['sharpness_p90'] == 120.0
        assert got['aesthetic_score'] == 5.5
        assert got['coherence_state'] == 'ok'


def test_a_corrupt_blob_does_not_stop_the_pass(app, bank, monkeypatch):
    with app.app_context():
        rows = clips_of(bank)
        rows[0].metrics_json = 'not json at all'
        db.session.commit()
        store = {rows[0].id: frames((0.25, vec(1, 0)), (2.25, vec(1, 0)))}
        monkeypatch.setattr(coherence, 'load_vectors', lambda _b: store)
        assert coherence.run_coherence(bank)['measured'] == 1


# --- the cut, derived at read time -------------------------------------------------------

def test_moving_the_cut_re_sorts_the_bank_with_no_rescan():
    """The doctrine, pinned: scores are stored, verdicts are derived. The same
    stored numbers answer differently under two different cuts."""
    scores = {'coherence_state': 'ok', 'coherence_span': 0.78,
              'coherence_min': 0.70, 'coherence_frames': 3}
    assert 'missed_cut' in video_metrics.verdicts(scores, {'coherence_floor': 0.80})
    assert 'missed_cut' not in video_metrics.verdicts(scores, {'coherence_floor': 0.75})


def test_no_cut_set_flags_nothing():
    """It ships empty, and empty means empty — not zero."""
    scores = {'coherence_state': 'ok', 'coherence_span': 0.10}
    assert video_metrics.verdicts(scores, {}) == set()
    assert video_metrics.verdicts(scores, {'coherence_floor': None}) == set()


def test_a_shot_with_no_reading_is_never_flagged():
    """No measurement, no verdict — the rule every cut in this lane obeys."""
    assert 'missed_cut' not in video_metrics.verdicts({}, {'coherence_floor': 0.99})
    assert 'missed_cut' not in video_metrics.verdicts(
        {'coherence_state': 'one_frame'}, {'coherence_floor': 0.99})


def test_the_default_config_ships_the_cut_empty():
    from app.config import DEFAULTS
    assert DEFAULTS['video_bank']['coherence_floor'] is None


# --- the contracts that keep the pass wired ------------------------------------------------

def test_every_key_this_pass_owns_survives_a_re_measure():
    """A key this writes that ADVISORY_KEYS does not carry is erased by the next
    quality scan, in silence, sending a whole bank back through a pass it has
    already paid for."""
    for key in coherence.OWNED_KEYS:
        assert key in video_metrics.ADVISORY_KEYS, key


def test_the_cut_is_declared_where_the_panel_reads_it():
    assert 'coherence_floor' in video_metrics.THRESHOLD_KEYS


def test_the_still_flag_is_left_to_the_pass_that_measures_motion():
    """The second flag Panda-70M's rule suggests is deliberately NOT shipped —
    measured, this number tracks shot LENGTH rather than motion (see the module
    docstring). This pins the decision: a perfectly self-similar shot raises
    NOTHING here, and stillness stays the metrics pass's word.

    A test on an absence, because the absence is the finding: re-adding a
    near-1 flag would be re-adding a measurement that was refuted."""
    perfect = {'coherence_state': 'ok', 'coherence_span': 1.0,
               'coherence_min': 1.0, 'coherence_frames': 3}
    flags = video_metrics.verdicts(perfect, {k: 0.9 for k
                                             in video_metrics.THRESHOLD_KEYS})
    assert 'still' not in flags
    assert 'slideshow' not in flags
    assert 'missed_cut' not in flags


def test_the_embed_pass_checks_coherence_when_it_is_done(app, bank, monkeypatch):
    """The step is reachable, or the whole feature is dead code. It runs AFTER
    the embedding — that is where the vectors it reads come from — and its
    numbers reach the job's own result."""
    from app.services import video_bank_service as svc
    with app.app_context():
        rows = clips_of(bank)
        store = {rows[0].id: frames((0.25, vec(1, 0)), (2.25, vec(0, 1)))}
        monkeypatch.setattr(coherence, 'load_vectors', lambda _b: store)
        job = {'done': 0, 'total': 4, 'detail': ''}
        out = svc._check_coherence(job, bank, False)
    # The store held one shot of four: the other three are REPORTED, not folded
    # into a total that would read as a clean run.
    assert out == {'coherence_measured': 1, 'coherence_unmeasured': 3}
    assert 'scene' in job['detail']


def test_it_runs_after_the_look_score_not_before(app):
    """Order, pinned. 🎨 pays a torch import and can fail on a machine with no
    egress; this pays dot products and cannot. Behind it, the cheap certainty
    still lands when the expensive one does not."""
    import inspect
    from app.services import video_bank_service as svc
    source = inspect.getsource(svc._embed_job)
    assert source.index('_rate_the_look') < source.index('_check_coherence')


def test_a_re_embed_asks_for_a_re_check(app, monkeypatch):
    """Rewritten vectors are different vectors — `reembed` IS `recheck`."""
    from app.services import video_bank_service as svc
    seen = {}
    monkeypatch.setattr(
        'app.services.video_temporal_coherence.pending_clips',
        lambda bank_id, recheck=False: [object()])
    monkeypatch.setattr(
        'app.services.video_temporal_coherence.run_coherence',
        lambda bank_id, recheck=False, **kw: seen.update(recheck=recheck)
        or {'measured': 0, 'one_frame': 0, 'unmeasured': 0})
    with app.app_context():
        svc._check_coherence({'done': 0, 'total': 0}, 1, True)
    assert seen == {'recheck': True}


def test_a_cancelled_embed_pass_does_not_run_the_check(app, monkeypatch):
    from app.services import video_bank_service as svc

    def _explode(*a, **k):
        raise AssertionError('the coherence check ran after a cancel')

    monkeypatch.setattr(
        'app.services.video_temporal_coherence.run_coherence', _explode)
    with app.app_context():
        assert svc._check_coherence({'done': 0, 'total': 0, 'cancelled': True},
                                    1, False) == {}


def test_an_embed_run_with_nothing_left_to_check_says_nothing(app, bank, monkeypatch):
    """The common case on a re-run: no announcement and no noise in the job's
    final sentence."""
    from app.services import video_bank_service as svc
    with app.app_context():
        rows = clips_of(bank)
        store = {r.id: frames((0.25, vec(1, 0)), (2.25, vec(1, 0))) for r in rows}
        monkeypatch.setattr(coherence, 'load_vectors', lambda _b: store)
        job = {'done': 0, 'total': 4, 'detail': 'embedding shots (CPU)'}
        svc._check_coherence(job, bank, False)
        assert svc._check_coherence(job, bank, False) == {}
    assert job['detail'] == 'checking each shot holds one scene'


def test_stillness_still_comes_from_motion_and_is_unaffected_by_coherence():
    """The adjacency, pinned from the other side: a shot that is perfectly
    coherent AND motionless is `still`, and a shot that is perfectly coherent and
    moving is not — neither answer moves when the coherence number does."""
    base = {'metrics_state': 'ok', 'coherence_state': 'ok', 'coherence_span': 1.0}
    cuts = {'motion_floor': 0.001}
    assert 'still' in video_metrics.verdicts({**base, 'motion_mean': 0.0001}, cuts)
    assert 'still' not in video_metrics.verdicts({**base, 'motion_mean': 0.01}, cuts)
