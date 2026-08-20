"""🎬 Type a word, get the scenes — the embedding pass and the search over it.

Two halves, and the interesting decisions are all in the seams between them.

THE PASS embeds SEVERAL frames per shot, not one. A shot is a span of time, and
a thumbnail is one instant of it: a car that only enters frame in the last second
is invisible to a search that embedded the opening frame, and there is no error
to see — the clip simply never comes back. So each clip contributes a frame near
its start, its ambassador frame (the sharpest one the metrics scan measured), and
one near its end, and a clip's score for a query is the MAX over those. That is
the whole reason this file exists rather than a two-line reuse of the image lane.

THE SEARCH reuses the image bank's text encoder untouched (one CLIP checkpoint,
one query cache, app-wide) and its honesty rules: a ranking is not a filter, a
clip with no embedding cannot be found by any phrase and that has to be said out
loud, and no similarity threshold is invented because the measurements say no
threshold separates relevant from unrelated.

Both heavy seams — the frame decode and the ML subprocess — are monkeypatched
here, so the suite stays green with neither PyAV nor torch installed.
"""
import pytest

from app.services import clip_text_encoder
from app.services import video_clip_search as vcs


# --- which frames of a shot get embedded ---------------------------------------

def test_a_shot_contributes_a_frame_at_each_end_and_its_ambassador():
    """The load-bearing choice. One frame per shot would make a scene findable
    only if its subject happens to be on screen at that instant."""
    times = vcs.frame_times(0.0, 10.0, sharpest_s=6.0)

    assert [label for label, _ in times] == ['start', 'key', 'end']
    assert dict(times)['key'] == 6.0


def test_the_end_frames_are_inset_from_the_cut():
    """A shot boundary is where a cut just happened: the very first and very last
    frames are disproportionately dissolves, black and half-faded — the same
    reason the thumbnail pass never grabs frame 0."""
    times = dict(vcs.frame_times(4.0, 14.0, sharpest_s=9.0))

    assert times['start'] > 4.0
    assert times['end'] < 14.0
    assert all(4.0 <= t <= 14.0 for t in times.values())


def test_a_shot_with_no_measured_ambassador_still_gets_a_middle_frame():
    """The metrics scan may not have run. Falling back to the middle keeps the
    three-frame contract instead of silently embedding two."""
    times = dict(vcs.frame_times(0.0, 8.0, sharpest_s=None))

    assert set(times) == {'start', 'key', 'end'}
    assert 3.0 < times['key'] < 5.0


def test_a_very_short_shot_is_not_embedded_three_times_over():
    """Half a second of footage has one frame's worth of information in it, and
    paying CLIP three times for the same picture is pure waste."""
    times = vcs.frame_times(0.0, 0.2, sharpest_s=0.1)

    assert len(times) == 1


def test_an_ambassador_outside_the_shot_is_ignored():
    """Bounds can be re-cut after measuring, which leaves a sharpest-frame
    timestamp pointing outside the shot it describes. Clamping it back inside
    would invent a measurement; dropping it falls back to the middle."""
    times = dict(vcs.frame_times(10.0, 20.0, sharpest_s=3.0))

    assert 10.0 < times['key'] < 20.0


# --- the pass ------------------------------------------------------------------

def test_every_clip_of_the_bank_gets_its_frames_embedded(app, monkeypatch):
    bank_id = _bank_with_clips(app, 3)
    _fake_seams(monkeypatch)

    with app.app_context():
        out = vcs.run_embed(bank_id)
        store = vcs.load_embeddings(bank_id)

    assert out['embedded'] == 3
    assert set(store) == set(_clip_ids(app, bank_id))
    assert all(len(frames) == 3 for frames in store.values())
    assert _states(app, bank_id) == ['ok', 'ok', 'ok']


def test_a_rerun_pays_only_for_what_the_first_run_had_not_reached(app, monkeypatch):
    """The resume contract every pass in this lane shares."""
    bank_id = _bank_with_clips(app, 3)
    seen = _fake_seams(monkeypatch)

    with app.app_context():
        vcs.run_embed(bank_id)
        first = len(seen)
        vcs.run_embed(bank_id)

    assert first == 3
    assert len(seen) == 3


def test_reembedding_redoes_everything(app, monkeypatch):
    bank_id = _bank_with_clips(app, 2)
    seen = _fake_seams(monkeypatch)

    with app.app_context():
        vcs.run_embed(bank_id)
        vcs.run_embed(bank_id, reembed=True)

    assert len(seen) == 4


def test_one_undecodable_shot_costs_that_shot_not_the_pass(app, monkeypatch):
    """A bank is embedded in bulk; a bitstream error in shot 200 must not throw
    away 199 vectors."""
    bank_id = _bank_with_clips(app, 3)
    calls = {'n': 0}

    def flaky(src_path, times, dest_dir, stem):
        calls['n'] += 1
        if calls['n'] == 2:
            raise OSError('bitstream error')
        return [(label, t, f'{dest_dir}/{stem}_{label}.jpg') for label, t in times]

    monkeypatch.setattr(vcs, '_write_frames', flaky)
    monkeypatch.setattr(vcs, '_encode_frame_files',
                        lambda paths, **kw: [_vec(1, 0) for _ in paths])

    with app.app_context():
        out = vcs.run_embed(bank_id)

    assert out['embedded'] == 2 and out['unreadable'] == 1
    assert sorted(_states(app, bank_id)) == ['ok', 'ok', 'unreadable']


# --- when it is the MACHINE failing, not the footage -----------------------------
# 'unreadable' retires a shot: `pending_clips` stops offering it, ✂ Duplicates and
# 🎨 the look score stop seeing it, and nothing in the app ever re-tests the guess.
# So the pass must not write that word about a failure it cannot attribute to the
# file — which is exactly what it did for 855 shots in a row while the frame
# encoder could not start at all.

def test_a_frame_encoder_that_cannot_start_marks_nothing_at_all(app, monkeypatch):
    """THE regression, in one test. The encoder raising is a statement about
    this interpreter, and the next shot would repeat it identically — so the
    pass stops instead of writing 861 verdicts about footage it never read."""
    from app.services.clip_image_encoder import ImageEncodeError

    bank_id = _bank_with_clips(app, 6)
    monkeypatch.setattr(vcs, '_write_frames',
                        lambda src, times, d, stem:
                        [(l, t, f'{d}/{stem}_{l}.jpg') for l, t in times])

    def dead(paths, **kw):
        raise ImageEncodeError("ML dependency installed but broken: importing it "
                               "raised AttributeError: module 'ssl' has no "
                               "attribute 'wrap_socket'")

    monkeypatch.setattr(vcs, '_encode_frame_files', dead)

    with app.app_context():
        out = vcs.run_embed(bank_id)

    assert out['embedded'] == 0
    assert out['unreadable'] == 0, 'no shot may be blamed for a dead worker'
    assert _states(app, bank_id) == [None] * 6
    # And the sentence carries the child's own words, not a generic phrase.
    assert 'wrap_socket' in out['aborted']


def test_a_run_of_failures_stops_the_pass_before_it_retires_them(app, monkeypatch):
    """The backstop, for an environment failure that arrives as something other
    than an encoder error — a decode library that cannot load, a bank folder
    that moved. 855 identical failures is not 855 facts about 855 files."""
    bank_id = _bank_with_clips(app, vcs.SYSTEMIC_STREAK + 5)

    def always_bad(src_path, times, dest_dir, stem):
        raise OSError('cannot open the decoder')

    monkeypatch.setattr(vcs, '_write_frames', always_bad)
    monkeypatch.setattr(vcs, '_encode_frame_files', lambda paths, **kw: [])

    with app.app_context():
        out = vcs.run_embed(bank_id)

    assert out['unreadable'] == 0
    assert set(_states(app, bank_id)) == {None}
    assert 'cannot open the decoder' in out['aborted']


def test_a_rough_patch_in_a_bank_is_still_only_a_rough_patch(app, monkeypatch):
    """The guard counts CONSECUTIVE failures and resets on every success, so a
    bank with bad files scattered through it never trips it — and each of those
    files still gets the verdict it earned."""
    bank_id = _bank_with_clips(app, 12)
    seen = {'n': 0}

    def every_other(src_path, times, dest_dir, stem):
        seen['n'] += 1
        if seen['n'] % 2 == 0:
            raise OSError('bitstream error')
        return [(l, t, f'{dest_dir}/{stem}_{l}.jpg') for l, t in times]

    monkeypatch.setattr(vcs, '_write_frames', every_other)
    monkeypatch.setattr(vcs, '_encode_frame_files',
                        lambda paths, **kw: [_vec(1, 0) for _ in paths])

    with app.app_context():
        out = vcs.run_embed(bank_id)

    assert out['aborted'] is None
    assert out['embedded'] == 6 and out['unreadable'] == 6


def test_a_tail_of_failures_shorter_than_the_streak_is_still_the_files_fault(
        app, monkeypatch):
    """Held verdicts are not lost verdicts. A bank whose last shots are damaged
    ends normally, and those shots are retired like any other bad file — the
    guard only fires on a run long enough to mean something."""
    bank_id = _bank_with_clips(app, 4)
    seen = {'n': 0}

    def last_two_bad(src_path, times, dest_dir, stem):
        seen['n'] += 1
        if seen['n'] > 2:
            raise OSError('bitstream error')
        return [(l, t, f'{dest_dir}/{stem}_{l}.jpg') for l, t in times]

    monkeypatch.setattr(vcs, '_write_frames', last_two_bad)
    monkeypatch.setattr(vcs, '_encode_frame_files',
                        lambda paths, **kw: [_vec(1, 0) for _ in paths])

    with app.app_context():
        out = vcs.run_embed(bank_id)

    assert out['aborted'] is None
    assert _states(app, bank_id) == ['ok', 'ok', 'unreadable', 'unreadable']


def test_a_pass_that_gave_up_leaves_every_shot_in_the_queue(app, monkeypatch):
    """The whole point of not marking them: the next click picks them up. A
    marked shot would be gone from `pending_clips` forever."""
    from app.services.clip_image_encoder import ImageEncodeError

    bank_id = _bank_with_clips(app, 4)
    monkeypatch.setattr(vcs, '_write_frames',
                        lambda src, times, d, stem:
                        [(l, t, f'{d}/{stem}_{l}.jpg') for l, t in times])
    broken = {'yes': True}

    def encode(paths, **kw):
        if broken['yes']:
            raise ImageEncodeError('the frame encoder produced no result')
        return [_vec(1, 0) for _ in paths]

    monkeypatch.setattr(vcs, '_encode_frame_files', encode)

    with app.app_context():
        vcs.run_embed(bank_id)
        assert vcs.pending_clips(bank_id).count() == 4
        broken['yes'] = False
        out = vcs.run_embed(bank_id)

    assert out['embedded'] == 4


# --- the way back, for a bank that was already retired ---------------------------

def test_find_scenes_picks_the_failed_shots_back_up_when_nothing_else_is_left(
        app, monkeypatch):
    """The dead end this closes: a bank whose shots were all marked 'unreadable'
    answered "0 shots to embed" to 🔎 Find scenes and "run Find scenes first" to
    ✂ Duplicates — at the same time, with no way out of the app.

    'unreadable' is a HYPOTHESIS about a file, formed by a pass that may have
    been wrong about its whole environment, and a hypothesis nothing ever
    re-tests becomes a fact by accident."""
    bank_id = _bank_with_clips(app, 3)
    _mark_all(app, bank_id, 'unreadable')

    with app.app_context():
        assert vcs.pending_clips(bank_id).count() == 3

    _fake_seams(monkeypatch)
    with app.app_context():
        out = vcs.run_embed(bank_id)

    assert out['embedded'] == 3
    assert _states(app, bank_id) == ['ok', 'ok', 'ok']


def test_a_bank_with_new_shots_does_not_re_decode_the_failed_ones(app):
    """The retry is the SECOND tier and only that: on the normal path — a bank
    that grew — it costs nothing, which is what lets it ride the button the user
    was going to click anyway instead of needing one of its own."""
    from app.extensions import db
    from app.models import VideoClip

    bank_id = _bank_with_clips(app, 3)
    _mark_all(app, bank_id, 'unreadable')
    with app.app_context():
        first = VideoClip.query.filter_by(bank_id=bank_id).first()
        db.session.add(VideoClip(bank_id=bank_id, source_id=first.source_id,
                                 start_s=99.0, end_s=104.0))
        db.session.commit()
        pending = vcs.pending_clips(bank_id).all()

    assert [c.embed_state for c in pending] == [None]


def test_an_embedded_shot_is_never_offered_again(app):
    """The resume contract the retry must not break."""
    bank_id = _bank_with_clips(app, 3)
    _mark_all(app, bank_id, 'ok')

    with app.app_context():
        assert vcs.pending_clips(bank_id).count() == 0


def test_the_job_says_the_pass_was_abandoned_and_runs_nothing_after_it(
        app, monkeypatch):
    """What the user actually reads. The look score and the coherence check both
    consume the vectors this pass did not produce, so letting them run would bury
    the one line that matters under two "0 shot(s) rated" — the same silence the
    original failure hid behind."""
    from app.services import video_bank_service as svc

    bank_id = _bank_with_clips(app, 2)
    monkeypatch.setattr(
        'app.services.video_clip_search.run_embed',
        lambda *a, **k: {'embedded': 0, 'unreadable': 0,
                         'aborted': 'the frame encoder could not run: boom'})

    def _explode(*a, **k):
        raise AssertionError('a follow-on pass ran after an abandoned embed')

    monkeypatch.setattr(svc, '_rate_the_look', _explode)
    monkeypatch.setattr(svc, '_check_coherence', _explode)
    job = {'done': 0, 'total': 0, 'detail': ''}
    monkeypatch.setattr(svc.bank_jobs, 'progress',
                        lambda j, **kw: j.update(kw))
    monkeypatch.setattr(svc.bank_jobs, 'cancelled', lambda j: False)

    with app.app_context():
        svc._embed_job(bank_id, False, False)(job)

    assert job['detail'].startswith('stopped —')
    assert 'the frame encoder could not run' in job['detail']


def _mark_all(app, bank_id, state):
    from app.extensions import db
    from app.models import VideoClip
    with app.app_context():
        for clip in VideoClip.query.filter_by(bank_id=bank_id).all():
            clip.embed_state = state
        db.session.commit()


def test_the_vector_store_survives_a_restart(app, monkeypatch):
    """Vectors live in a file next to the bank, not in memory: a pass that ran
    overnight must still be searchable tomorrow."""
    bank_id = _bank_with_clips(app, 2)
    _fake_seams(monkeypatch)

    with app.app_context():
        vcs.run_embed(bank_id)
        vcs.forget_memory_cache()
        store = vcs.load_embeddings(bank_id)

    assert len(store) == 2


def test_recutting_a_shot_forgets_what_it_used_to_look_like(app, monkeypatch):
    """The vectors describe THREE INSTANTS of a span. Move the span and they stop
    describing this shot — worse, they keep describing frames it no longer
    contains, so a search would return it for a car that is now in the
    neighbouring shot and point the player at a second outside its own bounds.
    Exactly the reason a re-cut already drops the thumbnail and the metrics."""
    from app.services import video_bank_service as svc
    bank_id = _bank_with_clips(app, 2)
    _fake_seams(monkeypatch)
    cid = _clip_ids(app, bank_id)[0]

    monkeypatch.setattr(clip_text_encoder, 'encode_query', lambda text: (_vec(1, 0), True))

    with app.app_context():
        vcs.run_embed(bank_id)
        svc.set_clip_bounds('local', bank_id, cid, 1.0, 4.0)
        found = vcs.search('local', bank_id, 'a red car')

    # The property, not a proxy for it: the re-cut shot is out of the ranking.
    assert cid not in found['clip_ids']
    assert found['unembedded'] == 1
    # The OTHER shot is untouched — a retouch costs the shot it retouched.
    assert len(found['clip_ids']) == 1


def test_the_next_pass_sweeps_away_the_vectors_of_a_shot_that_was_recut(app,
                                                                        monkeypatch):
    """A trim clears the STATE and leaves the vectors on disk on purpose — the
    store is tens of MB on a real bank and a trim is an interactive gesture. So
    the file has to be swept somewhere, and the only free place is a pass that
    was going to rewrite it anyway. Without this the store grows orphans forever
    on a bank that is trimmed often."""
    from app.services import video_bank_service as svc
    bank_id = _bank_with_clips(app, 2)
    _fake_seams(monkeypatch)
    cid = _clip_ids(app, bank_id)[0]

    with app.app_context():
        vcs.run_embed(bank_id)
        svc.set_clip_bounds('local', bank_id, cid, 1.0, 4.0)
        assert cid in vcs.load_embeddings(bank_id)     # still there, unreachable
        vcs.run_embed(bank_id)                          # re-embeds it under new bounds
        svc.set_clip_bounds('local', bank_id, cid, 2.0, 5.0)
        vcs.run_embed(bank_id, reembed=False)
        store = vcs.load_embeddings(bank_id)

    # Re-embedded under the new bounds, and its stale copy did not survive twice.
    assert len(store[cid]) == 3


# --- the search ----------------------------------------------------------------

def test_a_word_that_only_appears_at_the_end_of_a_shot_still_finds_it(app, monkeypatch):
    """THE test this design exists for. Clip A matches the query weakly for its
    whole duration; clip B is unrelated until its last second, where the thing
    being searched for walks into frame. Scoring a shot by ONE frame — or by the
    mean of its frames — buries B. The max does not."""
    bank_id = _bank_with_clips(app, 2)
    ids = _clip_ids(app, bank_id)
    query = _vec(1, 0)
    vectors = {
        ids[0]: [_mix(0.4), _mix(0.4), _mix(0.4)],       # mildly on-topic throughout
        ids[1]: [_mix(0.0), _mix(0.0), _vec(1, 0)],      # dead on, at the very end
    }
    _store(app, bank_id, vectors)
    monkeypatch.setattr(clip_text_encoder, 'encode_query', lambda text: (query, True))

    with app.app_context():
        out = vcs.search('local', bank_id, 'a red car')

    assert [r['clip_id'] for r in out['results']] == [ids[1], ids[0]]
    assert out['results'][0]['frame_label'] == 'end'


def test_a_result_says_which_second_of_the_shot_matched(app, monkeypatch):
    """The number the player needs to seek to. Without it the user is handed a
    30-second shot and told the answer is somewhere inside it."""
    bank_id = _bank_with_clips(app, 1)
    cid = _clip_ids(app, bank_id)[0]
    _store(app, bank_id, {cid: [_mix(0.1), _vec(1, 0), _mix(0.1)]})
    monkeypatch.setattr(clip_text_encoder, 'encode_query', lambda text: (_vec(1, 0), True))

    with app.app_context():
        out = vcs.search('local', bank_id, 'a red car')

    hit = out['results'][0]
    assert hit['frame_label'] == 'key'
    assert hit['frame_s'] == pytest.approx(vcs.frame_times(0.0, 5.0, None)[1][1], abs=0.6)


def test_shots_with_no_embedding_are_counted_out_loud(app, monkeypatch):
    """A shot that was never embedded cannot be found by ANY phrase. Answering
    "3 results" without saying so lets the user conclude the scene is not in the
    bank — which is how a search silently becomes a lie."""
    bank_id = _bank_with_clips(app, 3)
    cid = _clip_ids(app, bank_id)[0]
    _store(app, bank_id, {cid: [_vec(1, 0)]})
    monkeypatch.setattr(clip_text_encoder, 'encode_query', lambda text: (_vec(1, 0), True))

    with app.app_context():
        out = vcs.search('local', bank_id, 'a red car')

    assert out['pool'] == 1
    assert out['unembedded'] == 2


def test_searching_a_bank_that_was_never_embedded_says_what_to_run(app, monkeypatch):
    bank_id = _bank_with_clips(app, 2)
    monkeypatch.setattr(clip_text_encoder, 'encode_query', lambda text: (_vec(1, 0), True))

    with app.app_context():
        with pytest.raises(ValueError, match='Search'):
            vcs.search('local', bank_id, 'a red car')


def test_an_empty_query_is_refused_before_anything_is_encoded(app, monkeypatch):
    bank_id = _bank_with_clips(app, 1)
    calls = []
    monkeypatch.setattr(clip_text_encoder, 'encode_query',
                        lambda text: (calls.append(text), (_vec(1, 0), True))[1])

    with app.app_context():
        with pytest.raises(ValueError):
            vcs.search('local', bank_id, '   ')

    assert calls == []


def test_a_pushed_down_term_sinks_the_shots_that_look_like_it(app, monkeypatch):
    """CLIP ignores "without" — it does not negate, it silently returns the very
    thing that was excluded. The subtraction is the only mechanism that works, so
    the query grammar offers it as `-term`, exactly like the image lane."""
    bank_id = _bank_with_clips(app, 2)
    ids = _clip_ids(app, bank_id)
    # The first shot is the closest to the query AND the only one wearing a hat.
    _store(app, bank_id, {ids[0]: [_unit(0.90, 0.50, 0.0)],
                          ids[1]: [_unit(0.85, 0.00, 0.5)]})
    vectors = {'a red car': _unit(1, 0, 0), 'hat': _unit(0, 1, 0),
               'a hat': _unit(0, 1, 0)}
    monkeypatch.setattr(clip_text_encoder, 'encode_query',
                        lambda text: (vectors[text], True))

    with app.app_context():
        plain = vcs.search('local', bank_id, 'a red car')
        pushed = vcs.search('local', bank_id, 'a red car -hat')
        by_field = vcs.search('local', bank_id, 'a red car', push_down='a hat')

    assert [r['clip_id'] for r in plain['results']] == [ids[0], ids[1]]
    assert [r['clip_id'] for r in pushed['results']] == [ids[1], ids[0]]
    assert pushed['push_down'] == 'hat'
    # A multi-word trait cannot ride in `-term` (one whitespace token), so the
    # second field is the route for it — and both must mean the same thing.
    assert [r['clip_id'] for r in by_field['results']] == [ids[1], ids[0]]


def test_the_ranking_reports_the_pool_median_it_was_measured_against(app, monkeypatch):
    """No absolute score band exists for CLIP cosines, and inventing one is how a
    search either returns nothing on a perfect match or everything on a bad one.
    The only honest yardstick is what a TYPICAL shot of THIS bank scores for THIS
    phrase — measured, per query, never assumed."""
    bank_id = _bank_with_clips(app, 3)
    ids = _clip_ids(app, bank_id)
    _store(app, bank_id, {ids[0]: [_mix(0.9)], ids[1]: [_mix(0.5)], ids[2]: [_mix(0.1)]})
    monkeypatch.setattr(clip_text_encoder, 'encode_query', lambda text: (_vec(1, 0), True))

    with app.app_context():
        out = vcs.search('local', bank_id, 'a red car')

    assert out['pool_median'] is not None
    assert out['score_range']['top'] >= out['score_range']['bottom']


def test_results_carry_the_shot_row_the_grid_already_knows_how_to_draw(app, monkeypatch):
    """The ranking is an ORDER, and a second request to fetch the rows would lose
    it (the clip list endpoint sorts by file and start time, deliberately). So the
    ranked rows come back with the ranking."""
    bank_id = _bank_with_clips(app, 2)
    ids = _clip_ids(app, bank_id)
    _store(app, bank_id, {ids[0]: [_mix(0.2)], ids[1]: [_mix(0.9)]})
    monkeypatch.setattr(clip_text_encoder, 'encode_query', lambda text: (_vec(1, 0), True))

    with app.app_context():
        out = vcs.search('local', bank_id, 'a red car')

    assert [c['id'] for c in out['clips']] == [ids[1], ids[0]]
    assert 'duration_s' in out['clips'][0]


def test_n_caps_what_comes_back(app, monkeypatch):
    bank_id = _bank_with_clips(app, 4)
    ids = _clip_ids(app, bank_id)
    _store(app, bank_id, {cid: [_mix(0.5)] for cid in ids})
    monkeypatch.setattr(clip_text_encoder, 'encode_query', lambda text: (_vec(1, 0), True))

    with app.app_context():
        out = vcs.search('local', bank_id, 'a red car', n=2)

    assert len(out['results']) == 2 and out['pool'] == 4


def test_a_bank_that_is_not_yours_is_not_searchable(app, monkeypatch):
    bank_id = _bank_with_clips(app, 1)
    monkeypatch.setattr(clip_text_encoder, 'encode_query', lambda text: (_vec(1, 0), True))

    with app.app_context():
        with pytest.raises(ValueError, match='not found'):
            vcs.search('someone-else', bank_id, 'a red car')


# --- the real decode seam, when the extra is present ----------------------------

def test_real_decode_writes_one_readable_frame_per_requested_time(tmp_path):
    """Integration: the one test that runs the true PyAV path, on a synthesised
    clip. Skips — loudly — when the video extra is absent.

    It pins the two things the monkeypatched tests cannot see: a frame really
    lands on disk for each requested second, and it is written at the CLIP size
    rather than at the metrics scan's 160 px analysis size. Feeding an upscaled
    160 px frame to a model that sees 224×224 would degrade every vector in the
    bank to save nothing — the decode, not the resize, is what costs."""
    av = pytest.importorskip('av', reason='video extra not installed')
    Image = pytest.importorskip('PIL.Image', reason='Pillow not installed')
    import numpy as np

    path = str(tmp_path / 'clip.mp4')
    with av.open(path, 'w') as container:
        vs = container.add_stream('h264', rate=10)
        vs.width, vs.height, vs.pix_fmt = 640, 360, 'yuv420p'
        for i in range(40):
            img = np.full((360, 640, 3), (i * 6) % 255, dtype=np.uint8)
            for packet in vs.encode(av.VideoFrame.from_ndarray(img, format='rgb24')):
                container.mux(packet)
        for packet in vs.encode():
            container.mux(packet)

    times = vcs.frame_times(0.0, 3.5, sharpest_s=2.0)
    written = vcs._write_frames(path, times, str(tmp_path / 'frames'), 'clip_1')

    assert [label for label, _, _ in written] == [label for label, _ in times]
    for _label, _t, jpeg in written:
        with Image.open(jpeg) as im:
            assert max(im.size) == vcs.EMBED_LONG_SIDE
            assert min(im.size) > 1


def test_the_embedding_frames_are_bigger_than_the_analysis_frames():
    """Two passes, two sizes, on purpose — pinned as data so a refactor that
    'unifies' them has to argue with a test rather than with a comment."""
    from app.services import video_metrics_scan
    assert vcs.EMBED_LONG_SIDE > video_metrics_scan.ANALYSIS_WIDTH
    assert vcs.EMBED_LONG_SIDE >= 224


# --- helpers -------------------------------------------------------------------

def _unit(*components):
    import numpy as np
    v = np.asarray([float(c) for c in components], dtype='float32')
    return v / (float(np.linalg.norm(v)) + 1e-8)


def _vec(x, y):
    return _unit(x, y, 0.0)


def _mix(weight):
    """A unit vector `weight` of the way from "unrelated" to "exactly the query"."""
    return _vec(weight, 1.0 - weight)


def _fake_seams(monkeypatch):
    """Stand in for the PyAV decode and the CLIP subprocess. Returns the list the
    decode appends to, so a test can count what was really re-done."""
    seen = []

    def write(src_path, times, dest_dir, stem):
        seen.append(stem)
        return [(label, t, f'{dest_dir}/{stem}_{label}.jpg') for label, t in times]

    monkeypatch.setattr(vcs, '_write_frames', write)
    monkeypatch.setattr(vcs, '_encode_frame_files',
                        lambda paths, **kw: [_vec(1, 0) for _ in paths])
    return seen


def _store(app, bank_id, vectors):
    """Write a vector store directly — the search's input, without a pass."""
    with app.app_context():
        from app.extensions import db
        from app.models import VideoClip
        store = {}
        for cid, vecs in vectors.items():
            clip = db.session.get(VideoClip, cid)
            times = vcs.frame_times(clip.start_s, clip.end_s, None)
            store[cid] = [{'label': times[i % len(times)][0],
                           'time_s': times[i % len(times)][1], 'vec': v}
                          for i, v in enumerate(vecs)]
            clip.embed_state = 'ok'
        db.session.commit()
        vcs.save_embeddings(bank_id, store)
        vcs.forget_memory_cache()


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
        for i in range(n):
            db.session.add(VideoClip(bank_id=bank.id, source_id=src.id,
                                     start_s=float(i * 10), end_s=float(i * 10 + 5)))
        db.session.commit()
        return bank.id


def _clip_ids(app, bank_id):
    from app.models import VideoClip
    with app.app_context():
        return [c.id for c in VideoClip.query.filter_by(bank_id=bank_id)
                .order_by(VideoClip.id).all()]


def _states(app, bank_id):
    from app.models import VideoClip
    with app.app_context():
        return [c.embed_state for c in VideoClip.query.filter_by(bank_id=bank_id)
                .order_by(VideoClip.id).all()]
