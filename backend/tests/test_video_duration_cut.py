"""🎚 The duration cut — the one quality cut that needs no measurement pass.

TransNetV2 runs with `min_shot_frames=5` on purpose: a real flash cut is a real
shot, and a detector that refuses to emit one hides genuine boundaries. The price
is a triage grid peppered with half-second shots that can never reach a dataset —
`command_for_profile` refuses them at promotion — but that refusal happens at the
END, after the user has already scrolled past them a hundred times.

Two things separate this cut from the other eight and both are tested here:

  * duration is on the CLIP ROW, derived from the bounds, so it is known for
    every clip the moment it is detected. There is no "unmeasured" state for it,
    which means the cut works on a bank nobody has run the metrics pass over —
    and that is the state a user is in when the clutter first bothers them;
  * `brief` is not `too_short`. The promotion's `too_short` is an impossibility
    of the target profile; `brief` is a judgement the user sets. Same clip, two
    different sentences, and collapsing them would make a user think lowering the
    panel value buys them a dataset.
"""
import json

import pytest

from app.services import video_bank_service as svc

from _video_extra import detect_source_stub
from app.services import video_metrics as vm


# --- the rule ------------------------------------------------------------------

def test_a_clip_shorter_than_the_cut_is_flagged_brief():
    scores = {'metrics_state': 'ok', 'motion_mean': 0.5}

    assert 'brief' in vm.verdicts(scores, {'min_duration_s': 1.0}, duration_s=0.6)
    assert 'brief' not in vm.verdicts(scores, {'min_duration_s': 1.0}, duration_s=2.4)


def test_a_clip_exactly_at_the_cut_is_not_brief():
    """`below`, like every floor in this panel — a clip that meets the number the
    user typed is a clip they asked to keep."""
    assert vm.verdicts({}, {'min_duration_s': 1.0}, duration_s=1.0) == set()


def test_no_duration_cut_flags_nothing():
    """The panel ships with every cut empty and this one is no exception: a
    default here would flag a share of the bank nobody chose."""
    assert vm.verdicts({'metrics_state': 'ok'}, {}, duration_s=0.1) == set()


def test_the_brief_flag_does_not_wait_for_a_metrics_pass():
    """The whole point. Every other flag needs `run_metrics`; this one reads a
    number the detector already wrote, so it fires on a bank straight out of
    detection — with no scores dict at all."""
    assert vm.verdicts(None, {'min_duration_s': 1.0}, duration_s=0.6) == {'brief'}


def test_a_clip_with_no_known_duration_is_never_brief():
    """Same rule as every other cut: absence of a reading is not a defect."""
    assert vm.verdicts({}, {'min_duration_s': 1.0}) == set()


def test_the_dry_run_counts_the_duration_cut_per_rule():
    bank = [({'metrics_state': 'ok'}, 0.4),
            ({'metrics_state': 'ok'}, 0.9),
            ({'metrics_state': 'ok'}, 6.0)]

    preview = vm.dry_run(bank, {'min_duration_s': 1.0})

    assert preview['brief'] == 2
    assert preview['total_flagged'] == 2


def test_the_dry_run_still_accepts_a_plain_scores_list():
    """Callers that know nothing about duration keep working — they simply cannot
    raise `brief`, which is the honest answer for a caller with no bounds."""
    preview = vm.dry_run([{'metrics_state': 'ok', 'motion_mean': 0.0001}],
                         {'motion_floor': 0.001, 'min_duration_s': 1.0})

    assert preview['still'] == 1
    assert 'brief' not in preview


# --- it has to be reachable from the app ----------------------------------------

def test_the_duration_cut_is_readable_as_a_threshold(app):
    """A cut `verdicts()` honours but `metric_thresholds()` never reads exists
    only in the source — this lane has already paid for that once with
    `first_frame_floor`."""
    with app.app_context():
        assert 'min_duration_s' in svc.metric_thresholds()


@pytest.fixture()
def seams(monkeypatch):
    """A 120 s source cut into one 8 s shot and one 0.5 s flash — the shape the
    user complained about, with nothing measured on either."""
    monkeypatch.setattr(svc, '_probe_file', lambda _p: {
        'duration_s': 120.0, 'fps_native': 30.0, 'width': 1920, 'height': 1080,
        'codec': 'h264', 'probe_state': 'ok', 'file_size': 4096})
    monkeypatch.setattr(svc, '_detect_source', detect_source_stub([
        {'start_s': 0.0, 'end_s': 8.0, 'start_frame': 0, 'end_frame': 240},
        {'start_s': 41.25, 'end_s': 41.75, 'start_frame': 1237, 'end_frame': 1252}]))
    monkeypatch.setattr(svc, '_write_thumbnail', lambda *a, **k: True)


def _detected_bank(client, tmp_path):
    folder = tmp_path / 'rushes'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'a.mp4').write_bytes(b'\x00' * 32)
    r = client.post('/api/video-bank/create',
                    json={'name': 'rushes', 'folder': str(folder)})
    assert r.status_code == 200, r.get_json()
    bank_id = r.get_json()['id']
    assert client.post(f'/api/video-bank/{bank_id}/pipeline',
                       json={}).status_code == 202
    return bank_id


def test_the_flash_cut_is_flagged_in_the_grid_with_nothing_measured(
        client, tmp_path, seams, app):
    """End to end on the state that produced the complaint: detected, never
    measured. The 0.5 s shot carries the chip, the 8 s one does not."""
    bank_id = _detected_bank(client, tmp_path)
    assert client.put('/api/settings',
                      json={'config': {'video_bank': {'min_duration_s': 1.0}}}
                      ).status_code == 200

    rows = client.get(f'/api/video-bank/{bank_id}/clips').get_json()['clips']

    by_duration = {round(r['duration_s'], 2): r for r in rows}
    assert by_duration[0.5]['metrics'] is None, 'fixture measured something'
    assert by_duration[0.5]['flags'] == ['brief']
    assert by_duration[8.0]['flags'] == []


def test_the_dry_run_endpoint_previews_the_duration_cut(client, tmp_path, seams):
    """The preview is what keeps a mis-set cut from gutting a bank, so a cut the
    route drops reports zero and reassures. Asserted through the COUNT."""
    bank_id = _detected_bank(client, tmp_path)

    body = client.post(f'/api/video-bank/{bank_id}/metrics-dry-run',
                       json={'min_duration_s': 1.0}).get_json()

    assert body['brief'] == 1
    assert body['total_flagged'] == 1


def test_the_dry_run_sees_clips_the_metrics_pass_never_touched(
        client, tmp_path, seams, app):
    """The counting used to start from the clips carrying scores, which for a
    duration cut is the wrong population: the flash cuts are exactly the ones a
    user has not bothered measuring yet."""
    from app.extensions import db
    from app.models import VideoClip
    bank_id = _detected_bank(client, tmp_path)
    with app.app_context():
        measured = (VideoClip.query.filter_by(bank_id=bank_id)
                    .order_by(VideoClip.start_s.asc()).first())
        measured.metrics_json = json.dumps({'metrics_state': 'ok',
                                            'motion_mean': 0.5})
        db.session.commit()

    body = client.post(f'/api/video-bank/{bank_id}/metrics-dry-run',
                       json={'min_duration_s': 1.0}).get_json()

    assert body['brief'] == 1
