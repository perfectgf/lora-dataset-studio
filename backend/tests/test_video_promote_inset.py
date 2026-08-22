"""✂ Trimming the edges of every shot at promotion.

A shot boundary is where a cut just happened, so the frames immediately around it
are disproportionately dissolves, fades and leftovers of a transition. The
embedding pass already refuses to look at them (`video_clip_search.EDGE_MARGIN_S`
= 0.25 s), and the reason applies far more strongly to what actually gets
TRAINED on: a dataset whose clips all open on half a dissolve teaches the model
to open on half a dissolve.

So a promotion can inset both bounds. Three decisions carry it:

DEFAULT ZERO, ALWAYS. Turning this on by default would silently change what every
existing recipe exports — the same folder, the same name, different content, with
nothing to say so. It is asked for per promotion.

AN INSET NEVER SHORTENS A CLIP. The target's VAE takes a frame COUNT (4n+1 for
Wan, 8n+1 for LTX, 17n+5 for H3 — see video_targets.py), and `ClipTooShort`
exists because ffmpeg happily writes a 32-frame file and exits 0 when asked for
81 (video_clip_export.py:56-64). An inset that eats into the margin must
therefore DROP the clip, never export a short one — ai-toolkit would train it as
repeated stills without a word.

AND IT SAYS WHICH CLIPS IT COST. "Too short for 81 frames" and "too short ONCE
0.25 s was trimmed off each end" are the same sentence to a user and two
different problems: the first is a clip that was never usable, the second is one
the user's own setting removed, and only the second is fixed by lowering the
inset. Counting them together is how a setting quietly halves a dataset.
"""
import pytest

from app.extensions import db
from app.models import VideoClip, VideoSource
from app.services import video_bank_service as svc
from app.services import video_clip_export as export

LOCAL_USER = 'local'

# wan22_14b: 16 fps, 81 frames → 80/16 = 5.0 s of source needed.
PROFILE = 'wan22_14b'
FRAMES = 81
NEEDED_S = 5.0


@pytest.fixture()
def seams(monkeypatch):
    """ffmpeg replaced by a file-toucher, and the argv of every call recorded —
    the bounds actually asked for are the thing under test."""
    calls = []

    def _run(args):
        calls.append(list(args))
        with open(args[-1], 'wb') as fh:
            fh.write(b'\x00')
        return 0, ''
    monkeypatch.setattr(svc, '_run_ffmpeg', _run)
    monkeypatch.setattr(svc, '_ffmpeg_or_raise', lambda: '/usr/bin/ffmpeg')
    monkeypatch.setattr(svc, '_write_thumbnail', lambda *a, **k: True)
    return calls


# --- the arithmetic, on its own -------------------------------------------------

def test_a_span_that_supplies_the_frames_fits_and_one_that_does_not_says_no():
    assert export.fits_frames(5.0, FRAMES, 16) is True
    assert export.fits_frames(4.9, FRAMES, 16) is False
    # The same float slack `clip_command` already allows: bounds arrive as PTS
    # seconds and an exact 5.0 can present as 4.999999999999999.
    assert export.fits_frames(5.0 - 1e-9, FRAMES, 16) is True


def test_fits_frames_agrees_with_the_exception_it_predicts():
    """It exists to tell WHY a clip was refused, so it must never disagree with
    the refusal itself — a second implementation of the same rule that drifts is
    worse than no explanation at all."""
    for span in (4.0, 4.999, 5.0, 6.0):
        predicted = export.fits_frames(span, FRAMES, 16)
        try:
            export.clip_command(ffmpeg='ffmpeg', src='a', dst='b', start_s=0.0,
                                end_s=span, frames=FRAMES, fps=16)
            raised = False
        except export.ClipTooShort:
            raised = True
        assert predicted is not raised, f'disagreement at {span}s'


# --- the inset at promotion ------------------------------------------------------

def test_by_default_nothing_is_trimmed(app, tmp_path, seams):
    """The whole point of the default: an existing recipe exports exactly what it
    exported yesterday."""
    bank_id = _bank(app, tmp_path, spans=[(10.0, 20.0)])

    _promote(app, bank_id)

    assert _bounds(seams)[0] == pytest.approx(10.0)


def test_an_inset_moves_the_start_inwards(app, tmp_path, seams):
    bank_id = _bank(app, tmp_path, spans=[(10.0, 20.0)])

    _promote(app, bank_id, edge_inset_s=0.25)

    assert _bounds(seams)[0] == pytest.approx(10.25)


def test_the_tail_is_trimmed_too_and_not_only_the_head(app, tmp_path, seams):
    """A transition sits at BOTH boundaries, so both are trimmed. The end bound
    never appears in the argv — the command asks for a frame COUNT from `-ss`,
    deliberately (video_clip_export.py:24-28) — so the two-sided trim is asserted
    where it is observable: on the SPAN that remains available.

    5.25 s minus 0.25 s at each end is 4.75 s, which cannot supply 81 frames at
    16 fps. Trimming the head alone would leave 5.0 s and the clip would land."""
    bank_id = _bank(app, tmp_path, spans=[(0.0, 5.25)])

    result = _promote(app, bank_id, edge_inset_s=0.25)

    assert seams == []
    assert 'edge trim' in result['detail']


def test_a_clip_the_inset_makes_too_short_is_dropped_not_shortened(app, tmp_path,
                                                                   seams):
    """The load-bearing refusal. 5.2 s fits 81 frames; trimmed by 0.25 s at each
    end it is 4.7 s and does not. Exporting it anyway writes a file ai-toolkit
    trains as repeated stills, silently."""
    bank_id = _bank(app, tmp_path, spans=[(0.0, 5.2)])

    result = _promote(app, bank_id, edge_inset_s=0.25)

    assert seams == []                                   # nothing encoded
    assert '1 dropped by the 0.25s edge trim' in result['detail']
    assert 'too short' not in result['detail']


def test_a_clip_that_never_fitted_keeps_its_own_reason(app, tmp_path, seams):
    """4 s cannot supply 81 frames at 16 fps with or without an inset. Blaming
    the user's setting for it would send them to lower a knob that changes
    nothing."""
    bank_id = _bank(app, tmp_path, spans=[(0.0, 4.0)])

    result = _promote(app, bank_id, edge_inset_s=0.25)

    assert '1 too short for 81 frames' in result['detail']
    assert 'edge trim' not in result['detail']


def test_the_two_reasons_are_reported_side_by_side(app, tmp_path, seams):
    bank_id = _bank(app, tmp_path,
                    spans=[(0.0, 4.0), (10.0, 15.2), (20.0, 30.0)])

    result = _promote(app, bank_id, edge_inset_s=0.25)

    assert result['detail'] == ('done — 1 clips encoded, 1 too short for 81 '
                                'frames, 1 dropped by the 0.25s edge trim')
    assert len(seams) == 1


def test_the_promotion_says_up_front_what_the_inset_will_cost(app, tmp_path, seams):
    """Before a single file is written. The count is cheap (bounds and the
    profile's own arithmetic) and it is the difference between choosing an inset
    and discovering it."""
    bank_id = _bank(app, tmp_path, spans=[(0.0, 10.0), (20.0, 25.2)])

    result = _promote(app, bank_id, edge_inset_s=0.25)

    assert result['composition']['edge_inset_s'] == 0.25
    assert result['composition']['inset_would_drop'] == 1


def test_no_inset_reports_no_cost(app, tmp_path, seams):
    bank_id = _bank(app, tmp_path, spans=[(0.0, 10.0)])

    result = _promote(app, bank_id)

    assert result['composition']['edge_inset_s'] == 0.0
    assert result['composition']['inset_would_drop'] == 0


def test_an_inset_that_would_take_every_clip_is_still_the_users_call(app, tmp_path,
                                                                    seams):
    """Reported, not refused. The user may be about to lower it, and a promotion
    that returns "0 clips" while the preview said 0 would be the confusing one."""
    bank_id = _bank(app, tmp_path, spans=[(0.0, 5.2), (10.0, 15.2)])

    result = _promote(app, bank_id, edge_inset_s=0.25)

    assert result['composition']['inset_would_drop'] == 2
    assert seams == []


# --- refusals --------------------------------------------------------------------

def test_a_negative_inset_is_refused(app, tmp_path, seams):
    """It would EXTEND the clip past its own bounds, into the neighbouring shot —
    the exact frames the detector decided did not belong to it."""
    bank_id = _bank(app, tmp_path, spans=[(10.0, 20.0)])

    with app.app_context():
        with pytest.raises(ValueError, match='inset'):
            svc.start_promote(app, LOCAL_USER, bank_id, name='Set',
                              target_profile=PROFILE, frames=FRAMES,
                              edge_inset_s=-0.1)


def test_an_absurd_inset_is_refused_rather_than_silently_emptying_the_dataset(
        app, tmp_path, seams):
    bank_id = _bank(app, tmp_path, spans=[(10.0, 20.0)])

    with app.app_context():
        with pytest.raises(ValueError, match='inset'):
            svc.start_promote(app, LOCAL_USER, bank_id, name='Set',
                              target_profile=PROFILE, frames=FRAMES,
                              edge_inset_s=99.0)


def test_the_route_passes_the_inset_through(client, app, tmp_path, seams):
    """A parameter the service honours and the route drops is a setting that
    exists only in the tests — this lane already carries one (`max_per_source`,
    implemented and unreachable over HTTP)."""
    bank_id = _bank(app, tmp_path, spans=[(10.0, 20.0)])

    r = client.post(f'/api/video-bank/{bank_id}/promote',
                    json={'name': 'Set', 'target_profile': PROFILE,
                          'frames': FRAMES, 'edge_inset_s': 0.25})

    assert r.status_code == 202
    assert r.get_json()['composition']['edge_inset_s'] == 0.25
    assert _bounds(seams)[0] == pytest.approx(10.25)


def test_a_bad_inset_over_http_is_a_400(client, app, tmp_path, seams):
    bank_id = _bank(app, tmp_path, spans=[(10.0, 20.0)])

    r = client.post(f'/api/video-bank/{bank_id}/promote',
                    json={'name': 'Set', 'target_profile': PROFILE,
                          'frames': FRAMES, 'edge_inset_s': -1})

    assert r.status_code == 400


# --- helpers ---------------------------------------------------------------------

def _bank(app, tmp_path, spans):
    folder = tmp_path / 'rushes'
    folder.mkdir(exist_ok=True)
    (folder / 's0.mp4').write_bytes(b'\x00')
    with app.app_context():
        bank, _ = svc.create_bank(LOCAL_USER, 'rushes', str(folder))
        bank_id = bank.id
        src = VideoSource.query.filter_by(bank_id=bank_id).first()
        src.probe_state = 'ok'
        src.duration_s = 600.0
        src.fps_native = 30.0
        for start, end in spans:
            db.session.add(VideoClip(bank_id=bank_id, source_id=src.id,
                                     status='keep', start_s=start, end_s=end))
        db.session.commit()
        return bank_id


def _promote(app, bank_id, **kwargs):
    """Promote, and return the pre-flight result plus the job's closing SENTENCE.

    The sentence rather than the job's return value on purpose: `bank_jobs.start`
    discards what the runner returns (bank_jobs.py:49-58), and `detail` is what
    the user is actually told. Under TESTING the job runs inline, so it has
    already finished when start_promote returns."""
    from app.services import bank_jobs
    with app.app_context():
        result = svc.start_promote(app, LOCAL_USER, bank_id, name='Set',
                                   target_profile=PROFILE, frames=FRAMES,
                                   **kwargs)
        job = bank_jobs.get(svc.job_key(bank_id)) or {}
    return {**result, 'detail': job.get('detail') or ''}


def _bounds(calls):
    """The `-ss` value of every ffmpeg call, in order."""
    return [float(c[c.index('-ss') + 1]) for c in calls]


