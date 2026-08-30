"""🎬 What a promotion is MADE OF — composition reporting and the per-source cap.

The hole this closes was found by our own end-to-end test: it promoted "the first
50 clips that pass the filters", which meant "in id order", which meant
concentrated on the first few sources. Nothing limited one source's share of a
dataset, so a 50-clip dataset could quietly be 3 videos over-represented — the
kind of imbalance that overfits a source and that nobody sees, because the folder
looks exactly like a diverse one.

Two answers, both at promotion time:
  * the result REPORTS composition (sources, the biggest source's share), so the
    imbalance is at least visible;
  * an optional `max_per_source` cap spreads the selection, keeping the earliest
    clips of each source (earliest = detector order, stable and explainable).
"""
import pytest

from app.extensions import db
from app.models import VideoClip, VideoDataset, VideoSource
from app.services import video_bank_service as svc

LOCAL_USER = 'local'


@pytest.fixture()
def seams(monkeypatch):
    def _run(args):
        with open(args[-1], 'wb') as fh:
            fh.write(b'\x00')
        return 0, ''
    monkeypatch.setattr(svc, '_run_ffmpeg', _run)
    monkeypatch.setattr(svc, '_ffmpeg_or_raise', lambda: '/usr/bin/ffmpeg')
    monkeypatch.setattr(svc, '_write_thumbnail', lambda *a, **k: True)


def _bank_with_kept_clips(app, tmp_path, per_source=(6, 3, 1)):
    """A bank with len(per_source) probed sources and the given number of KEPT
    clips each, all long enough for 81 frames at 16 fps."""
    folder = tmp_path / 'rushes'
    folder.mkdir()
    for i in range(len(per_source)):
        (folder / f's{i}.mp4').write_bytes(b'\x00')
    with app.app_context():
        bank, _ = svc.create_bank(LOCAL_USER, 'rushes', str(folder))
        bank_id = bank.id
        sources = (VideoSource.query.filter_by(bank_id=bank_id)
                   .order_by(VideoSource.relpath).all())
        for src, count in zip(sources, per_source):
            src.probe_state = 'ok'
            src.duration_s = 600.0
            src.fps_native = 30.0
            for k in range(count):
                db.session.add(VideoClip(
                    bank_id=bank_id, source_id=src.id, status='keep',
                    start_s=float(k * 20), end_s=float(k * 20 + 10)))
        db.session.commit()
        return bank_id


def test_the_promotion_result_reports_its_composition(app, tmp_path, seams):
    """60% of this dataset comes from one source. The folder on disk looks
    exactly like a diverse one — the report is the only place the imbalance is
    visible before training on it."""
    bank_id = _bank_with_kept_clips(app, tmp_path, per_source=(6, 3, 1))

    with app.app_context():
        result = svc.start_promote(app, LOCAL_USER, bank_id, name='Set',
                                   target_profile='wan22_14b', frames=81)

    assert result['composition']['sources'] == 3
    assert result['composition']['top_source_share'] == pytest.approx(0.6)


def test_the_per_source_cap_spreads_the_selection(app, tmp_path, seams):
    """With a cap of 2, the 6-clip source contributes 2, not 6 — and the tiny
    source still contributes its single clip. The cap trims dominance, it never
    punishes scarcity."""
    bank_id = _bank_with_kept_clips(app, tmp_path, per_source=(6, 3, 1))

    with app.app_context():
        result = svc.start_promote(app, LOCAL_USER, bank_id, name='Set',
                                   target_profile='wan22_14b', frames=81,
                                   max_per_source=2)

    assert result['clips'] == 5                    # 2 + 2 + 1
    assert result['composition']['top_source_share'] == pytest.approx(2 / 5)


def test_the_cap_keeps_the_earliest_clips_of_each_source(app, tmp_path, seams):
    """Detector order, stable and explainable — not a random sample that changes
    on every promotion of the same bank."""
    bank_id = _bank_with_kept_clips(app, tmp_path, per_source=(4,))

    with app.app_context():
        result = svc.start_promote(app, LOCAL_USER, bank_id, name='Set',
                                   target_profile='wan22_14b', frames=81,
                                   max_per_source=2)
        ds = VideoDataset.query.filter_by(name='Set').one()
        from app.models import VideoDatasetClip
        starts = sorted(c.start_s for c in
                        VideoDatasetClip.query.filter_by(dataset_id=ds.id))

    assert result['clips'] == 2
    assert starts == [0.0, 20.0]                   # the two earliest


def test_no_cap_changes_nothing(app, tmp_path, seams):
    bank_id = _bank_with_kept_clips(app, tmp_path, per_source=(6, 3, 1))

    with app.app_context():
        result = svc.start_promote(app, LOCAL_USER, bank_id, name='Set',
                                   target_profile='wan22_14b', frames=81)

    assert result['clips'] == 10


def test_a_nonsense_cap_is_refused(app, tmp_path, seams):
    bank_id = _bank_with_kept_clips(app, tmp_path)

    with app.app_context():
        with pytest.raises(ValueError):
            svc.start_promote(app, LOCAL_USER, bank_id, name='Set',
                              target_profile='wan22_14b', frames=81,
                              max_per_source=0)


# --- the cap has to be REACHABLE (wave 4 finishing) ---------------------------
# Everything above was true and untestable from the app: `max_per_source` was
# honoured by the service and passed by neither the route nor the dialog, so the
# cap existed only in this file. A parameter no caller can send is not a feature,
# and it is the same shape of gap as `first_frame_floor` (a threshold the backend
# honoured and no reader named). These pin the whole path instead of the service.

def _bank_over_http(client, app, tmp_path, per_source=(6, 3, 1)):
    bank_id = _bank_with_kept_clips(app, tmp_path, per_source=per_source)
    return bank_id


def test_the_route_carries_the_cap_to_the_service(client, app, tmp_path, seams):
    bank_id = _bank_over_http(client, app, tmp_path)

    r = client.post(f'/api/video-bank/{bank_id}/promote',
                    json={'name': 'Set', 'target_profile': 'wan22_14b',
                          'frames': 81, 'max_per_source': 2})

    assert r.status_code == 202
    assert r.get_json()['clips'] == 5                       # 2 + 2 + 1


def test_no_cap_over_http_still_takes_everything(client, app, tmp_path, seams):
    """The default must not change: an existing recipe posts no cap and gets the
    whole selection, exactly as before."""
    bank_id = _bank_over_http(client, app, tmp_path)

    r = client.post(f'/api/video-bank/{bank_id}/promote',
                    json={'name': 'Set', 'target_profile': 'wan22_14b',
                          'frames': 81})

    assert r.get_json()['clips'] == 10


def test_a_null_cap_is_read_as_no_cap_and_not_as_zero(client, app, tmp_path, seams):
    """A form that clears the field sends null, and `int(None)` would be a 500."""
    bank_id = _bank_over_http(client, app, tmp_path)

    r = client.post(f'/api/video-bank/{bank_id}/promote',
                    json={'name': 'Set', 'target_profile': 'wan22_14b',
                          'frames': 81, 'max_per_source': None})

    assert r.status_code == 202
    assert r.get_json()['clips'] == 10


@pytest.mark.parametrize('bad', [0, -3, 'lots', 2.5])
def test_a_cap_that_is_not_a_whole_number_of_clips_is_a_named_400(
        client, app, tmp_path, seams, bad):
    """Named, not an `int()` traceback rendered as a message. 2.5 is in the list
    on purpose: "two and a half clips per source" is not a rounding question, it
    is a request nobody can mean."""
    bank_id = _bank_over_http(client, app, tmp_path)

    r = client.post(f'/api/video-bank/{bank_id}/promote',
                    json={'name': 'Set', 'target_profile': 'wan22_14b',
                          'frames': 81, 'max_per_source': bad})

    assert r.status_code == 400
    assert 'per source' in r.get_json()['error']


def test_a_capped_promotion_reports_the_composition_it_achieved(client, app,
                                                                tmp_path, seams):
    """The cap and the report are two halves of one answer: the share is what
    tells you whether the cap you chose was the right one."""
    bank_id = _bank_over_http(client, app, tmp_path)

    r = client.post(f'/api/video-bank/{bank_id}/promote',
                    json={'name': 'Set', 'target_profile': 'wan22_14b',
                          'frames': 81, 'max_per_source': 2})

    assert r.get_json()['composition']['top_source_share'] == pytest.approx(2 / 5)


def test_high_fps_sources_are_counted_never_judged(app, tmp_path, seams):
    """48+ fps footage is very often slow motion once conformed, and slo-mo
    teaches floaty movement (fal audited two thirds of their people clips as
    slo-mo). No detector is pretended: the count states the fps the probe
    already measured and lets the user weigh footage they know better than we
    do. The 30 fps fixture sources contribute zero."""
    bank_id = _bank_with_kept_clips(app, tmp_path, per_source=(2, 2))
    with app.app_context():
        slow = (VideoSource.query.filter_by(bank_id=bank_id)
                .order_by(VideoSource.relpath).first())
        slow.fps_native = 60.0
        db.session.commit()
        out = svc.start_promote(app, LOCAL_USER, bank_id,
                                name='mix', target_profile='wan22_14b',
                                frames=17)
        assert out['composition']['high_fps_clips'] == 2

