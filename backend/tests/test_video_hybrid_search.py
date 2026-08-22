"""🔎 Hybrid search — CLIP finds what is VISIBLE, captions find what HAPPENS.

The two halves answer different questions and neither replaces the other. CLIP
embeds frames, so it ranks by what a moment looks like; it cannot find "turns and
walks away", because that is a fact about time and no frame carries it. A caption
carries it and nothing else — it is a literal string, so it misses everything the
writer did not name.

HOW THEY ARE COMBINED, AND THE ONE NUMBER IN IT THAT IS NOT MEASURED.
The blend is `clip_similarity + w · caption_hit · spread`, where `caption_hit` is
the share of the query's words present in the caption and `spread` is THIS pool's
own gap between its best CLIP score and its median for THIS query.

That last factor is the point. CLIP cosines live in a narrow, per-bank band
(0.09-0.23 measured on this checkpoint), while a caption hit is 0..1. Adding them
raw would make any literal match outrank every visual one by a factor of five —
not a blend, a caption filter wearing a blend's clothes. Scaling the caption term
by the ranking's OWN spread makes the two commensurable without inventing a
conversion constant, which is the same reasoning that put `pool_median` in the
CLIP-only path.

`w` = 0.6 is INHERITED BY ANALOGY, not measured for this blend — the 0.6 that was
measured (image_bank_service.PUSH_DOWN_WEIGHT_DEFAULT, 7 316 images) is the
weight of a SUBTRACTED excluded phrase, a different experiment. It is a named,
tunable constant and it is reported to the caller so the UI never presents it as
a calibrated figure. This is stated here rather than in a commit message because
it is the one claim in this feature a reader should not take on trust.
"""

from app.services import clip_text_encoder
from app.services import video_clip_search as vcs

# The video-extra gate answers for the MACHINE, so without this these route
# tests pass where PyAV/ffmpeg are installed and 503 where they are not.
# Imported for its autouse effect; see _video_extra.py for why not importorskip.
from _video_extra import video_extra_ready  # noqa: F401


def _unit(*c):
    import numpy as np
    v = np.asarray([float(x) for x in c], dtype='float32')
    return v / (float(np.linalg.norm(v)) + 1e-8)


# --- the caption side, on its own ---------------------------------------------

def test_a_caption_hit_is_the_share_of_the_query_words_it_contains():
    """Share, not a boolean. "a red car" fully present is a stronger claim than
    one word of three, and collapsing them would rank a caption containing only
    "a" level with an exact match."""
    assert vcs.caption_hit('a woman walks away', 'woman walks') == 1.0
    assert vcs.caption_hit('a woman walks away', 'woman drives') == 0.5
    assert vcs.caption_hit('a woman walks away', 'car engine') == 0.0


def test_matching_ignores_case_punctuation_and_the_words_that_carry_nothing():
    """"The" and "a" appear in nearly every caption; counting them would give a
    free half-match to any query written as a sentence."""
    assert vcs.caption_hit('A woman turns, and walks away.', 'the woman TURNS') == 1.0
    assert vcs.caption_hit('A car.', 'a the of') == 0.0        # nothing but stopwords


def test_a_clip_with_no_caption_scores_nothing_rather_than_being_excluded():
    """It is still findable by CLIP. Treating "no caption" as "no match" would
    delete every un-captioned shot from every hybrid ranking, silently."""
    assert vcs.caption_hit(None, 'woman') == 0.0
    assert vcs.caption_hit('', 'woman') == 0.0


# --- the blend -----------------------------------------------------------------

def test_a_caption_match_lifts_a_shot_CLIP_alone_would_have_buried(app, monkeypatch):
    """The whole reason this exists. Two shots look equally like the query to
    CLIP; one's caption says the action out loud."""
    bank_id = _bank(app, 2)
    ids = _clip_ids(app, bank_id)
    _store(app, bank_id, {ids[0]: [_unit(0.9, 0.4)], ids[1]: [_unit(0.9, 0.4)]})
    _caption(app, ids[1], 'a woman turns and walks away')
    monkeypatch.setattr(clip_text_encoder, 'encode_query',
                        lambda text: (_unit(1, 0), True))

    with app.app_context():
        out = vcs.search('local', bank_id, 'woman walks away')

    assert out['results'][0]['clip_id'] == ids[1]
    assert out['hybrid'] is True


def test_a_caption_match_does_not_simply_outrank_everything(app, monkeypatch):
    """A blend, not a filter in disguise. A shot CLIP is far more confident about
    must still be able to win against a weak literal match — otherwise the visual
    half stops contributing the moment any caption exists."""
    bank_id = _bank(app, 2)
    ids = _clip_ids(app, bank_id)
    # Clip A: a much better visual match. Clip B: a partial caption hit.
    _store(app, bank_id, {ids[0]: [_unit(1, 0)], ids[1]: [_unit(0.2, 1.0)]})
    _caption(app, ids[1], 'a woman drives away')
    monkeypatch.setattr(clip_text_encoder, 'encode_query',
                        lambda text: (_unit(1, 0), True))

    with app.app_context():
        out = vcs.search('local', bank_id, 'woman walks')

    assert out['results'][0]['clip_id'] == ids[0]


def test_a_bank_with_no_captions_ranks_exactly_as_it_did_before(app, monkeypatch):
    """No silent change to a bank that never ran the caption pass."""
    bank_id = _bank(app, 3)
    ids = _clip_ids(app, bank_id)
    _store(app, bank_id, {ids[0]: [_unit(0.2, 1)], ids[1]: [_unit(0.9, 0.4)],
                          ids[2]: [_unit(0.6, 0.8)]})
    monkeypatch.setattr(clip_text_encoder, 'encode_query',
                        lambda text: (_unit(1, 0), True))

    with app.app_context():
        out = vcs.search('local', bank_id, 'woman walks')

    assert out['hybrid'] is False
    assert [r['clip_id'] for r in out['results']] == [ids[1], ids[2], ids[0]]


def test_the_answer_says_what_it_leaned_on_and_how_hard(app, monkeypatch):
    """The readiness line has to be able to say "CLIP only" or "CLIP + captions",
    and the weight is reported because it is NOT a measured constant — presenting
    it as one would be the dishonest part."""
    bank_id = _bank(app, 2)
    ids = _clip_ids(app, bank_id)
    _store(app, bank_id, {ids[0]: [_unit(1, 0)], ids[1]: [_unit(0.5, 0.9)]})
    _caption(app, ids[0], 'a woman walks away')
    monkeypatch.setattr(clip_text_encoder, 'encode_query',
                        lambda text: (_unit(1, 0), True))

    with app.app_context():
        out = vcs.search('local', bank_id, 'woman walks')

    assert out['hybrid'] is True
    assert out['caption_weight'] == vcs.HYBRID_CAPTION_WEIGHT
    assert out['captioned'] == 1
    assert out['results'][0]['caption_hit'] == 1.0


def test_a_result_carries_the_caption_so_the_grid_can_show_why(app, monkeypatch):
    """A ranking that moved a shot up for a reason the user cannot see is a
    ranking they cannot trust."""
    bank_id = _bank(app, 1)
    cid = _clip_ids(app, bank_id)[0]
    _store(app, bank_id, {cid: [_unit(1, 0)]})
    _caption(app, cid, 'a woman turns and walks away')
    monkeypatch.setattr(clip_text_encoder, 'encode_query',
                        lambda text: (_unit(1, 0), True))

    with app.app_context():
        out = vcs.search('local', bank_id, 'woman walks')

    assert out['clips'][0]['caption'] == 'a woman turns and walks away'


# --- promotion ------------------------------------------------------------------

def test_a_promoted_clip_carries_its_caption_into_the_sidecar(app, tmp_path,
                                                              monkeypatch):
    bank_id, dst = _promoted(app, tmp_path, monkeypatch,
                             caption='A woman walks away.')

    assert (dst.parent / (dst.stem + '.txt')).read_text(encoding='utf-8') \
        == 'A woman walks away.'


def test_a_clip_with_no_caption_still_gets_its_empty_sidecar(app, tmp_path,
                                                            monkeypatch):
    """The sidecar is ALWAYS written — musubi-tuner raises FileNotFoundError out
    of a worker future when it is missing, and diffusion-pipe drops the clip.
    Unchanged behaviour, pinned so captions cannot break it."""
    bank_id, dst = _promoted(app, tmp_path, monkeypatch, caption=None)

    assert (dst.parent / (dst.stem + '.txt')).read_text(encoding='utf-8') == ''


def test_the_promotion_says_up_front_how_many_clips_have_no_caption(app, tmp_path,
                                                                   monkeypatch):
    """An empty sidecar trains as an EMPTY PROMPT and ai-toolkit says nothing
    about it. That is exactly the kind of limit that has to be visible before the
    encode, not discovered in a training run."""
    from app.services import video_bank_service as svc
    # The assertion is about the COUNTS this promotion announces BEFORE it
    # encodes, so the encode is stubbed - the same two seams every other
    # promotion test stubs (test_video_promote_composition, _inset). Without
    # them the real cutter looks for ffmpeg on the machine and raises: green
    # where the video extra is installed, red on CI. The figures under test are
    # computed before a single frame is written.
    def _fake_ffmpeg(args):
        with open(args[-1], 'wb') as fh:
            fh.write(bytes(1))
        return 0, ''
    monkeypatch.setattr(svc, '_ffmpeg_or_raise', lambda: '/usr/bin/ffmpeg')
    monkeypatch.setattr(svc, '_run_ffmpeg', _fake_ffmpeg)
    bank_id = _bank(app, 3, keep=True)
    ids = _clip_ids(app, bank_id)
    _caption(app, ids[0], 'A woman walks away.')

    with app.app_context():
        result = svc.start_promote(app, 'local', bank_id, name='Set',
                                   target_profile='wan22_14b', frames=81)

    assert result['composition']['uncaptioned'] == 2
    assert result['composition']['captioned'] == 1


# --- helpers ---------------------------------------------------------------------

def _bank(app, n, keep=False):
    from app.extensions import db
    from app.models import VideoBank, VideoClip, VideoSource
    with app.app_context():
        bank = VideoBank(name='b', source_path='/srv/rushes')
        db.session.add(bank)
        db.session.flush()
        src = VideoSource(bank_id=bank.id, relpath='a.mp4', duration_s=600.0,
                          fps_native=30.0, probe_state='ok')
        db.session.add(src)
        db.session.flush()
        for i in range(n):
            db.session.add(VideoClip(bank_id=bank.id, source_id=src.id,
                                     start_s=float(i * 20), end_s=float(i * 20 + 10),
                                     status='keep' if keep else 'pending'))
        db.session.commit()
        return bank.id


def _clip_ids(app, bank_id):
    from app.models import VideoClip
    with app.app_context():
        return [c.id for c in VideoClip.query.filter_by(bank_id=bank_id)
                .order_by(VideoClip.id).all()]


def _caption(app, clip_id, text):
    from app.extensions import db
    from app.models import VideoClip
    with app.app_context():
        clip = db.session.get(VideoClip, clip_id)
        clip.caption = text
        clip.caption_state = 'ok'
        db.session.commit()


def _store(app, bank_id, vectors):
    from app.extensions import db
    from app.models import VideoClip
    with app.app_context():
        store = {}
        for cid, vecs in vectors.items():
            clip = db.session.get(VideoClip, cid)
            times = vcs.frame_times(clip.start_s, clip.end_s, None)
            store[cid] = [{'label': times[0][0], 'time_s': times[0][1], 'vec': v}
                          for v in vecs]
            clip.embed_state = 'ok'
        db.session.commit()
        vcs.save_embeddings(bank_id, store)
        vcs.forget_memory_cache()

def _promoted(app, tmp_path, monkeypatch, caption):
    """Run a one-clip promotion with ffmpeg faked, and return the written file.

    monkeypatch ONLY — a direct module assignment here leaks the fake into every
    test that runs after it in the same process, which this project has already
    paid for once."""
    from pathlib import Path
    from app.extensions import db
    from app.models import VideoClip, VideoSource
    from app.services import video_bank_service as svc

    folder = tmp_path / 'rushes'
    folder.mkdir(exist_ok=True)
    (folder / 's0.mp4').write_bytes(b'\x00')
    calls = []

    def _run(args):
        calls.append(list(args))
        with open(args[-1], 'wb') as fh:
            fh.write(b'\x00')
        return 0, ''
    monkeypatch.setattr(svc, '_run_ffmpeg', _run)
    monkeypatch.setattr(svc, '_ffmpeg_or_raise', lambda: '/usr/bin/ffmpeg')
    with app.app_context():
        bank, _ = svc.create_bank('local', 'rushes', str(folder))
        bank_id = bank.id
        src = VideoSource.query.filter_by(bank_id=bank_id).first()
        src.probe_state = 'ok'
        src.duration_s = 600.0
        src.fps_native = 30.0
        clip = VideoClip(bank_id=bank_id, source_id=src.id, status='keep',
                         start_s=0.0, end_s=10.0, caption=caption,
                         caption_state='ok' if caption else None)
        db.session.add(clip)
        db.session.commit()
        svc.start_promote(app, 'local', bank_id, name='Set',
                          target_profile='wan22_14b', frames=81)
    return bank_id, Path(calls[0][-1])
