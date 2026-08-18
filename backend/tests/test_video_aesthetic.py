"""🎨 The look score — the LAION aesthetic head over the vectors 🔎 Search cached.

What is under test is NOT "does the MLP work" (it is someone else's checkpoint and
it cannot even be imported in this interpreter) but the five decisions that make
this safe to run at the tail of somebody's embedding pass:

  * the AGGREGATION. Three frames become one number by a MEAN, and the choice is
    argued rather than defaulted — a max would collapse onto the ambassador frame
    and quietly re-report sharpness, a min would report the shot's worst boundary;
  * a shot with no vectors is UNMEASURED, never 0.0. Zero on a 1..10 scale is the
    strongest possible claim that a shot is hideous, and it is exactly what an
    install with no ✨ Score environment would otherwise write over a whole bank;
  * the verdict is derived at READ time, so moving `aesthetic_floor` re-sorts the
    bank with nothing re-scanned;
  * the pass MERGES into the shared metrics blob, and a re-measure keeps what it
    wrote — the failure mode ADVISORY_KEYS exists for;
  * the head not being available is a RESULT, not an exception: it rides an
    embedding run that succeeded, and reporting that run as failed because a
    13 MB download did not come back would be a lie about hours of work.
"""
import json

import pytest

from app.services import video_aesthetic as look
from app.services import video_metrics


# --- the aggregation --------------------------------------------------------------

def test_a_shot_is_rated_by_the_mean_of_its_frames():
    assert video_metrics.aesthetic_of([4.0, 6.0, 5.0]) == pytest.approx(5.0)


def test_a_shot_with_no_frame_scores_is_unrated_rather_than_hideous():
    """None, never 0.0. On a 1..10 scale a zero is a measurement — the harshest
    one available — and the absence of a measurement is not one."""
    assert video_metrics.aesthetic_of([]) is None
    assert video_metrics.aesthetic_of(None) is None


def test_the_rating_is_neither_the_best_frame_nor_the_worst():
    """The whole argument for the mean, pinned. A shot whose 'key' frame (the
    metrics pass's sharpest, and so usually its prettiest) rates 8 while an edge
    frame near the cut rates 2 must not come back as either: the max would make
    this a second reading of sharpness_p90, the min would rate the boundary."""
    value = video_metrics.aesthetic_of([2.0, 8.0, 5.0])
    assert value == pytest.approx(5.0)
    assert value != max([2.0, 8.0, 5.0])
    assert value != min([2.0, 8.0, 5.0])


def test_a_single_frame_shot_is_rated_on_that_frame():
    """Shots under MIN_SPAN_FOR_THREE_S get ONE embedded frame, and one reading
    is still a reading."""
    assert video_metrics.aesthetic_of([6.25]) == pytest.approx(6.25)


# --- the verdict ------------------------------------------------------------------

def test_a_shot_below_the_floor_is_flagged():
    flags = video_metrics.verdicts({'aesthetic_score': 3.4},
                                   {'aesthetic_floor': 4.0})
    assert 'low_aesthetic' in flags


def test_a_shot_above_the_floor_is_not():
    flags = video_metrics.verdicts({'aesthetic_score': 5.1},
                                   {'aesthetic_floor': 4.0})
    assert 'low_aesthetic' not in flags


def test_an_unrated_shot_is_never_flagged_whatever_the_floor():
    """"Not evaluated" must not read as "ugly". A bank half-rated would otherwise
    report its unrated half as the worst footage in it."""
    assert 'low_aesthetic' not in video_metrics.verdicts({'metrics_state': 'ok'},
                                                         {'aesthetic_floor': 10.0})


def test_with_no_floor_set_nothing_is_flagged():
    """No default cut, like every footage cut in this lane."""
    assert 'low_aesthetic' not in video_metrics.verdicts({'aesthetic_score': 1.0}, {})


def test_moving_the_floor_re_sorts_the_bank_with_nothing_re_scanned(app, monkeypatch):
    """The doctrine of the whole lane: raw scores are stored, verdicts are not.
    The blob is read once and asked twice — the flag appears and disappears with
    the cut, and the stored bytes never move."""
    bank_id, ids = _bank_with_clips(app, 1)
    _fake_scores(monkeypatch, {ids[0]: [3.0, 4.0, 5.0]})
    with app.app_context():
        look.run_aesthetic(bank_id)

    stored = _summaries(app, bank_id)[ids[0]]
    before = json.dumps(stored, sort_keys=True)
    assert 'low_aesthetic' in video_metrics.verdicts(stored, {'aesthetic_floor': 4.5})
    assert 'low_aesthetic' not in video_metrics.verdicts(stored, {'aesthetic_floor': 3.5})
    assert json.dumps(_summaries(app, bank_id)[ids[0]], sort_keys=True) == before


# --- the cut is reachable from both codebases -------------------------------------

def test_the_look_cut_is_one_the_panel_and_the_route_both_know():
    """THE canonical list. A cut that exists only in `verdicts()` is settable by
    hand-editing config.json and by nothing else — the exact failure
    `first_frame_floor` shipped with."""
    assert 'aesthetic_floor' in video_metrics.THRESHOLD_KEYS


def test_the_cut_ships_with_no_number_and_is_still_readable_as_a_setting():
    """Empty, like every cut that describes MATERIAL rather than a classifier's
    calibrated probability — and present in DEFAULTS, or a saved value would not
    survive the config merge."""
    from app.config import DEFAULTS
    assert DEFAULTS['video_bank']['aesthetic_floor'] is None


def test_the_threshold_reader_hands_the_cut_through(app):
    from app.services import video_bank_service as svc
    with app.app_context():
        assert 'aesthetic_floor' in svc.metric_thresholds()


# --- the shared blob ---------------------------------------------------------------

def test_re_measuring_a_bank_keeps_the_look_score():
    """The metrics scan rewrites metrics_json wholesale, and it decodes frames —
    it knows nothing about CLIP and could not recompute this. Without the merge,
    "measure again" would silently send the whole bank back through the aesthetic
    pass, with nothing to see but flags that stopped appearing."""
    merged = video_metrics.merge_advisory(
        {'metrics_state': 'ok', 'sharpness_p90': 3.0, 'aesthetic_score': 5.5},
        {'metrics_state': 'ok', 'sharpness_p90': 99.0})
    assert merged['aesthetic_score'] == 5.5
    assert merged['sharpness_p90'] == 99.0


def test_the_pass_keeps_the_measurements_it_did_not_write(app, monkeypatch):
    bank_id, ids = _bank_with_clips(app, 1)
    _measured(app, {ids[0]: 12.0})
    _fake_scores(monkeypatch, {ids[0]: [5.0]})

    with app.app_context():
        look.run_aesthetic(bank_id)

    stored = _summaries(app, bank_id)[ids[0]]
    assert stored['sharpness_p90'] == 12.0
    assert stored['metrics_state'] == 'ok'
    assert stored['aesthetic_score'] == pytest.approx(5.0)


# --- the pass over a bank -----------------------------------------------------------

def test_the_pass_rates_every_embedded_shot(app, monkeypatch):
    bank_id, ids = _bank_with_clips(app, 2)
    _fake_scores(monkeypatch, {ids[0]: [4.0, 6.0], ids[1]: [7.0]})

    with app.app_context():
        out = look.run_aesthetic(bank_id)

    assert out == {'rated': 2, 'unrated': 0, 'error': None}
    stored = _summaries(app, bank_id)
    assert stored[ids[0]]['aesthetic_score'] == pytest.approx(5.0)
    assert stored[ids[1]]['aesthetic_score'] == pytest.approx(7.0)


def test_a_shot_with_no_embeddings_is_unrated_rather_than_zero(app, monkeypatch):
    """The degraded case this lane cares about most: the embed pass never ran on
    this shot (or this install has no ✨ Score environment at all). It must come
    back with NO key — not a 0, and not a key set to None, either of which would
    retire it as "rated" and keep it out of the next run's queue forever."""
    bank_id, ids = _bank_with_clips(app, 2)
    with app.app_context():
        from app.extensions import db
        from app.models import VideoClip
        db.session.get(VideoClip, ids[1]).embed_state = None
        db.session.commit()
    _fake_scores(monkeypatch, {ids[0]: [5.0]})

    with app.app_context():
        out = look.run_aesthetic(bank_id)

    assert out['rated'] == 1
    stored = _summaries(app, bank_id)
    assert 'aesthetic_score' not in stored[ids[1]]
    assert stored[ids[1]] == {}
    assert 'low_aesthetic' not in video_metrics.verdicts(stored[ids[1]],
                                                        {'aesthetic_floor': 10.0})


def test_a_shot_the_store_does_not_hold_stays_in_the_queue(app, monkeypatch):
    """The column says 'ok' and the store disagrees — an interrupted flush, a
    store pruned under a live bank. Writing None would retire the shot as
    "rated, unrateable"; leaving it bare is what puts it back in the next run."""
    bank_id, ids = _bank_with_clips(app, 2)
    _fake_scores(monkeypatch, {ids[0]: [5.0]})

    with app.app_context():
        out = look.run_aesthetic(bank_id)
        assert out['unrated'] == 1
        assert [c.id for c in look.pending_clips(bank_id)] == [ids[1]]


def test_a_second_run_rates_nothing_and_never_starts_the_model(app, monkeypatch):
    """The resume contract, and the reason re-clicking 🔎 Find scenes on a rated
    bank is free: the whole subprocess (a cold `import torch`) is skipped when
    every embedded shot already carries a score."""
    bank_id, ids = _bank_with_clips(app, 1)
    _fake_scores(monkeypatch, {ids[0]: [5.0]})
    with app.app_context():
        look.run_aesthetic(bank_id)

    def _explode(*a, **k):
        raise AssertionError('the model was started with nothing to rate')

    monkeypatch.setattr(look, 'score_frames', _explode)
    with app.app_context():
        assert look.run_aesthetic(bank_id) == {'rated': 0, 'unrated': 0,
                                               'error': None}


def test_a_re_embed_re_rates_because_the_vectors_are_new(app, monkeypatch):
    """`reembed` rewrites every vector. A rating left beside new vectors would be
    a verdict about footage that has moved."""
    bank_id, ids = _bank_with_clips(app, 1)
    _fake_scores(monkeypatch, {ids[0]: [5.0]})
    with app.app_context():
        look.run_aesthetic(bank_id)
    _fake_scores(monkeypatch, {ids[0]: [8.0]})

    with app.app_context():
        out = look.run_aesthetic(bank_id, rescore=True)

    assert out['rated'] == 1
    assert _summaries(app, bank_id)[ids[0]]['aesthetic_score'] == pytest.approx(8.0)


def test_a_head_that_cannot_load_is_a_result_and_not_a_failed_pass(app, monkeypatch):
    """It rides an embedding run that SUCCEEDED. A machine with no egress cannot
    fetch the 13 MB head, and that must not report hours of decoding as failed —
    nor write a score nobody computed."""
    bank_id, ids = _bank_with_clips(app, 1)
    monkeypatch.setattr(look, 'score_frames',
                        lambda bank_id, **kw: (_ for _ in ()).throw(
                            RuntimeError('URLError: no route to host')))

    with app.app_context():
        out = look.run_aesthetic(bank_id)

    assert out['rated'] == 0
    assert 'no route to host' in out['error']
    assert _summaries(app, bank_id)[ids[0]] == {}


def test_the_pass_never_changes_a_triage_decision(app, monkeypatch):
    """Advisory means advisory: this writes a number, it does not reject."""
    bank_id, ids = _bank_with_clips(app, 2)
    _fake_scores(monkeypatch, {ids[0]: [1.0], ids[1]: [9.0]})

    with app.app_context():
        from app.models import VideoClip
        look.run_aesthetic(bank_id)
        assert {c.status for c in VideoClip.query.filter_by(bank_id=bank_id)} \
            == {'pending'}


def test_a_re_cut_shot_is_not_rated_on_the_vectors_of_its_old_span(app, monkeypatch):
    """A trim clears ``embed_state`` and deliberately leaves the .npz alone, so
    the store still holds three instants of a span that no longer exists. The
    COLUMN is the authority — the same rule ✂ Duplicates keeps."""
    bank_id, ids = _bank_with_clips(app, 1)
    with app.app_context():
        from app.extensions import db
        from app.models import VideoClip
        db.session.get(VideoClip, ids[0]).embed_state = None
        db.session.commit()
    _fake_scores(monkeypatch, {ids[0]: [5.0]})

    with app.app_context():
        assert look.run_aesthetic(bank_id) == {'rated': 0, 'unrated': 0,
                                               'error': None}
    assert _summaries(app, bank_id)[ids[0]] == {}


# --- the wiring: it rides 🔎 Find scenes ---------------------------------------------

def test_the_embed_pass_rates_the_look_when_it_is_done(app, monkeypatch):
    """The step is reachable, or the whole feature is dead code. It runs AFTER
    the embedding — that is where the vectors it reads come from — and its
    numbers reach the job's own result."""
    from app.services import video_bank_service as svc
    bank_id, ids = _bank_with_clips(app, 1)
    _fake_scores(monkeypatch, {ids[0]: [4.0, 6.0]})

    job = {'done': 0, 'total': 1, 'detail': ''}
    with app.app_context():
        out = svc._rate_the_look(job, bank_id, False)

    # `unrated` rides along since the wave's verification: a store missing half
    # its vectors must not read as a clean run (run_aesthetic's own docstring).
    assert out == {'rated': 1, 'unrated': 0, 'aesthetic_error': None}
    assert 'look' in job['detail']
    assert _summaries(app, bank_id)[ids[0]]['aesthetic_score'] == pytest.approx(5.0)


def test_a_re_embed_asks_for_a_re_rating(app, monkeypatch):
    """Rewritten vectors need rewritten ratings — `reembed` IS `rescore`."""
    from app.services import video_bank_service as svc
    seen = {}
    monkeypatch.setattr(
        'app.services.video_aesthetic.pending_clips',
        lambda bank_id, rescore=False: [object()])
    monkeypatch.setattr(
        'app.services.video_aesthetic.run_aesthetic',
        lambda bank_id, rescore=False, **kw: seen.update(rescore=rescore)
        or {'rated': 0, 'unrated': 0, 'error': None})

    with app.app_context():
        svc._rate_the_look({'done': 0, 'total': 0}, 1, True)
    assert seen == {'rescore': True}


def test_a_cancelled_embed_pass_is_not_charged_a_model_load(app, monkeypatch):
    """Stop means stop. A stopped pass has already kept everything it earned,
    and paying a cold `import torch` on the way out is not what the button says."""
    from app.services import video_bank_service as svc

    def _explode(*a, **k):
        raise AssertionError('the look score ran after a cancel')

    monkeypatch.setattr('app.services.video_aesthetic.run_aesthetic', _explode)
    with app.app_context():
        assert svc._rate_the_look({'done': 0, 'total': 0, 'cancelled': True},
                                  1, False) == {}


def test_an_embed_run_with_nothing_left_to_rate_says_nothing(app, monkeypatch):
    """The common case on a re-run: no announcement, no subprocess, no noise in
    the job's final sentence."""
    from app.services import video_bank_service as svc
    bank_id, ids = _bank_with_clips(app, 1)
    _fake_scores(monkeypatch, {ids[0]: [5.0]})
    job = {'done': 0, 'total': 1, 'detail': 'embedding shots (CPU)'}
    with app.app_context():
        svc._rate_the_look(job, bank_id, False)
        assert svc._rate_the_look(job, bank_id, False) == {}
    assert job['detail'] == 'rating how each shot looks'


# --- the worker's borrowed contracts ------------------------------------------------

def test_the_worker_reads_the_arrays_the_embed_pass_writes(app):
    """The store's schema is owned by ``video_clip_search.save_embeddings`` and
    merely BORROWED by the worker. A round trip through the real writer is what
    keeps the two from drifting — a renamed array would otherwise surface as a
    bank that silently rates nothing.

    The worker is read, not imported: it hides CUDA at import time (deliberately
    — see below), and doing that to the process running the suite would be a
    side effect on every test after it."""
    np = pytest.importorskip('numpy')
    from app.services import video_clip_search as vcs

    bank_id, ids = _bank_with_clips(app, 1)
    with app.app_context():
        vcs.save_embeddings(bank_id, {
            ids[0]: [{'label': 'key', 'time_s': 1.0,
                      'vec': np.zeros(768, dtype='float32')}]})
        path = vcs.embed_cache_path(bank_id)
        with np.load(str(path), allow_pickle=False) as z:
            for name in _worker_store_arrays():
                assert name in z.files, \
                    f'the embed pass no longer writes {name}'


def test_the_worker_rates_with_the_head_that_matches_the_encoder():
    """The invisible failure this lane is most exposed to. The LAION head was
    fitted on ONE embedding space — openai ViT-L/14, 768-d — and any other 768-d
    vector feeds it without error and yields a plausible, meaningless rating. So
    the head is imported from ``bank_score_infer`` rather than copied, and the
    encoder that produced the vectors being rated has to be that same pair."""
    import re
    worker = _infer_source('video_aesthetic_infer.py')
    assert 'import bank_score_infer' in worker, \
        'the video head must be the image lane\'s, not a second copy'
    assert '_load_aesthetic_head' in worker

    score = _infer_source('bank_score_infer.py')
    expects = re.search(r"_AESTHETIC_EXPECTS = \('([^']+)', '([^']+)'\)", score)
    assert expects, 'could not find _AESTHETIC_EXPECTS in bank_score_infer.py'
    frames = _infer_source('clip_image_embed_infer.py')
    name = re.search(r"^MODEL_NAME\s*=\s*'([^']+)'", frames, re.M)
    pre = re.search(r"^PRETRAINED\s*=\s*'([^']+)'", frames, re.M)
    assert (name.group(1), pre.group(1)) == expects.groups(), \
        ('the video frames are embedded by a checkpoint the aesthetic head was '
         'not fitted on — every rating would be plausible and meaningless')


def test_the_worker_never_takes_the_card():
    """This is one small matmul. Hiding CUDA before torch is imported is what
    stops a bank being rated from competing with a training run."""
    src = _infer_source('video_aesthetic_infer.py')
    hide = src.index("os.environ['CUDA_VISIBLE_DEVICES'] = ''")
    assert hide < src.index('import torch', hide), \
        'CUDA must be hidden before torch is imported'


# --- helpers -------------------------------------------------------------------------

def _infer_source(name):
    from pathlib import Path
    return (Path(__file__).resolve().parents[1] / 'infer'
            / name).read_text(encoding='utf-8')


def _worker_store_arrays():
    """``STORE_ARRAYS`` as the worker declares it, read off the source for the
    reason the docstring above gives."""
    import re
    m = re.search(r"^STORE_ARRAYS = \(([^)]*)\)", _infer_source(
        'video_aesthetic_infer.py'), re.M)
    assert m, 'could not find STORE_ARRAYS in video_aesthetic_infer.py'
    return [x for x in re.findall(r"'([a-z_]+)'", m.group(1))]


def _bank_with_clips(app, n):
    from app.extensions import db
    from app.models import VideoBank, VideoClip, VideoSource
    with app.app_context():
        bank = VideoBank(name='b', source_path='/srv/rushes')
        db.session.add(bank)
        db.session.flush()
        src = VideoSource(bank_id=bank.id, relpath='a.mp4', duration_s=60.0,
                          fps_native=25.0, probe_state='ok')
        db.session.add(src)
        db.session.flush()
        ids = []
        for i in range(n):
            # embed_state 'ok': the column is the AUTHORITY over the store, so a
            # fixture that leaves it null is a bank nothing can be rated in.
            clip = VideoClip(bank_id=bank.id, source_id=src.id,
                             start_s=float(i * 10), end_s=float(i * 10 + 5),
                             embed_state='ok')
            db.session.add(clip)
            db.session.flush()
            ids.append(clip.id)
        db.session.commit()
        return bank.id, ids


def _fake_scores(monkeypatch, per_clip):
    """The MODEL seam, stubbed. Nothing here imports torch or downloads a head:
    this pass reads vectors 🔎 Search already wrote and applies someone else's
    checkpoint to them."""
    monkeypatch.setattr(look, 'score_frames', lambda bank_id, **kw: dict(per_clip))


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
