"""🎬 The video lane's HTTP surface — it must feel like the image lane's.

A user does not know there are two services behind the app, and the seam is not
supposed to be visible. So these tests pin the shapes the image bank already
established, on the video routes: 202 for a pass that runs in the background, 409
carrying `busy_kind` when the bank is occupied, 404 for an unknown bank, 400 for a
refusal the user can fix, 503 for a missing tool.

Two of them are not about symmetry at all:

  * the blueprints have to be REGISTERED. `routes/__init__` imports them by name
    from a tuple, and a module that is written but not listed there answers 404
    everywhere while looking perfectly correct in the diff;
  * editing a caption has to rewrite the .txt on disk. The trainer never reads our
    database — it reads the sidecar next to the .mp4. A caption saved to one and
    not the other is a dataset that trains on the previous text while the UI shows
    the new one, with nothing anywhere to reveal it.

No ffmpeg, no PyAV, no torch: the four media seams are monkeypatched.
"""
import os

import pytest

from app.services import video_bank_service as svc

# The video-extra gate answers for the MACHINE, so without this these route
# tests pass where PyAV/ffmpeg are installed and 503 where they are not.
# Imported for its autouse effect; see _video_extra.py for why not importorskip.
from _video_extra import detect_source_stub, video_extra_ready  # noqa: F401


@pytest.fixture()
def seams(monkeypatch):
    calls = []

    def _run(args):
        calls.append(list(args))
        with open(args[-1], 'wb') as fh:
            fh.write(b'\x00')
        return 0, ''

    monkeypatch.setattr(svc, '_probe_file', lambda _p: {
        'duration_s': 120.0, 'fps_native': 30.0, 'width': 1920, 'height': 1080,
        'codec': 'h264', 'probe_state': 'ok', 'file_size': 4096})
    monkeypatch.setattr(svc, '_detect_source', detect_source_stub([
        {'start_s': 0.0, 'end_s': 8.0, 'start_frame': 0, 'end_frame': 240},
        {'start_s': 41.25, 'end_s': 50.0, 'start_frame': 1237, 'end_frame': 1500}]))
    monkeypatch.setattr(svc, '_write_thumbnail', lambda *a, **k: True)
    monkeypatch.setattr(svc, '_run_ffmpeg', _run)
    monkeypatch.setattr(svc, '_ffmpeg_or_raise', lambda: '/usr/bin/ffmpeg')
    return calls


def _folder(tmp_path, names=('a.mp4',)):
    folder = tmp_path / 'rushes'
    folder.mkdir(parents=True, exist_ok=True)
    for n in names:
        (folder / n).write_bytes(b'\x00' * 32)
    return str(folder)


def _make_bank(client, tmp_path, names=('a.mp4',)):
    r = client.post('/api/video-bank/create',
                    json={'name': 'rushes', 'folder': _folder(tmp_path, names)})
    assert r.status_code == 200, r.get_json()
    return r.get_json()['id']


def _ready_bank(client, tmp_path):
    """A bank scanned, detected and fully kept — the state promotion starts from."""
    bank_id = _make_bank(client, tmp_path)
    assert client.post(f'/api/video-bank/{bank_id}/pipeline',
                       json={}).status_code == 202
    assert client.post(f'/api/video-bank/{bank_id}/triage',
                       json={'ids': [], 'status': 'keep'}).status_code == 200
    return bank_id


# --- the blueprints exist and are wired ---------------------------------------

def test_the_video_bank_blueprint_is_registered(client, tmp_path):
    """`routes/__init__` imports blueprints by NAME from a tuple and swallows the
    ImportError of one that does not exist yet. A module written but not added to
    that tuple therefore answers 404 on every route while looking finished."""
    assert client.get('/api/video-banks').status_code == 200


def test_the_video_dataset_blueprint_is_registered(client):
    assert client.get('/api/video-datasets').status_code == 200


# --- creating and reading a bank ----------------------------------------------

def test_creating_a_bank_reports_what_it_inventoried(client, tmp_path):
    r = client.post('/api/video-bank/create',
                    json={'name': 'rushes', 'folder': _folder(tmp_path,
                                                              ('a.mp4', 'b.MOV'))})

    assert r.status_code == 200
    assert r.get_json()['added'] == 2


def test_a_folder_that_does_not_exist_is_a_400_not_a_500(client, tmp_path):
    """The most common first click in this lane is a pasted path with a typo."""
    r = client.post('/api/video-bank/create',
                    json={'name': 'x', 'folder': str(tmp_path / 'nope')})

    assert r.status_code == 400
    assert 'error' in r.get_json()


def test_the_workspace_payload_carries_counters_sources_and_capability(
        client, tmp_path, seams):
    """Capability rides on the payload as THREE booleans, not one verdict: decode,
    detect and encode fail independently and are fixed differently, so a single
    "video unavailable" is how a user reinstalls the wrong thing."""
    bank_id = _make_bank(client, tmp_path)

    body = client.get(f'/api/video-bank/{bank_id}').get_json()

    assert body['counts']['sources'] == 1
    assert body['sources'][0]['relpath'] == 'a.mp4'
    assert set(body['capability']) >= {'ok', 'decode', 'detect', 'encode'}


def test_an_unknown_bank_is_a_404(client):
    assert client.get('/api/video-bank/9999').status_code == 404


# --- the passes ----------------------------------------------------------------

@pytest.mark.parametrize('path', ['probe', 'detect', 'thumbs', 'pipeline'])
def test_every_pass_answers_202_and_runs_in_the_background(client, tmp_path,
                                                           seams, path):
    """A pass over a folder of rushes takes minutes. Holding the HTTP request open
    for it is not an option, so the POST returns immediately and the UI polls the
    bank payload — the same contract as every image-bank pass."""
    bank_id = _make_bank(client, tmp_path)

    r = client.post(f'/api/video-bank/{bank_id}/{path}', json={})

    assert r.status_code == 202, r.get_json()


@pytest.mark.parametrize('path', ['probe', 'detect', 'thumbs', 'pipeline'])
def test_a_busy_bank_refuses_with_the_kind_that_holds_it(client, tmp_path, seams,
                                                         path):
    """`busy_kind` is the machine-readable half. The refusal often lands before the
    first progress poll, so at that instant the response body is the only thing on
    the client that knows which pass is in the way — and parsing our own English
    sentence would be one rename away from breaking."""
    import time
    from app.services import bank_jobs
    bank_id = _make_bank(client, tmp_path)
    bank_jobs._jobs[svc.job_key(bank_id)] = {
        'kind': 'detect', 'done': 3, 'total': 9, 'error': None, 'cancelled': False,
        'finished': False, 'detail': None, 'started_at': time.time(),
        '_touched': time.time(), '_cancel_hook': None, 'pipeline': None}

    r = client.post(f'/api/video-bank/{bank_id}/{path}', json={})

    assert r.status_code == 409, r.get_json()
    assert r.get_json()['busy_kind'] == 'detect'


def test_a_pass_on_an_unknown_bank_is_a_404_not_a_400(client):
    """"Bank not found" is not a validation error the user can fix by editing the
    body — it means the bank was deleted in another tab."""
    assert client.post('/api/video-bank/9999/probe', json={}).status_code == 404


# --- triage --------------------------------------------------------------------

def test_triage_marks_the_clips_and_returns_the_new_counts(client, tmp_path, seams):
    """The counters ride back on the response so the gallery updates without a
    second round trip — a triage click is the most repeated gesture in this lane."""
    bank_id = _make_bank(client, tmp_path)
    client.post(f'/api/video-bank/{bank_id}/pipeline', json={})
    ids = client.get(f'/api/video-bank/{bank_id}/clips?ids_only=1').get_json()['ids']

    r = client.post(f'/api/video-bank/{bank_id}/triage',
                    json={'ids': ids[:1], 'status': 'reject', 'reason': 'blurry'})

    assert r.status_code == 200
    assert r.get_json()['counts']['reject'] == 1


def test_an_unknown_triage_status_is_refused(client, tmp_path, seams):
    """Only three words exist. A typo'd status silently writing itself into the
    column would make a clip invisible to every filter."""
    bank_id = _make_bank(client, tmp_path)

    r = client.post(f'/api/video-bank/{bank_id}/triage',
                    json={'ids': [], 'status': 'maybe'})

    assert r.status_code == 400


def test_the_clip_list_pages_and_filters(client, tmp_path, seams):
    bank_id = _make_bank(client, tmp_path)
    client.post(f'/api/video-bank/{bank_id}/pipeline', json={})

    body = client.get(f'/api/video-bank/{bank_id}/clips?status=pending').get_json()

    assert body['total'] == 2
    assert body['clips'][0]['start_s'] == 0.0
    assert body['clips'][0]['relpath'] == 'a.mp4'


def test_a_missing_thumbnail_is_a_404_rather_than_a_broken_image(client, tmp_path,
                                                                 seams):
    """The gallery renders a placeholder on 404. A 500 here would fill the console
    with errors for the ordinary case of a thumbnail pass that has not run."""
    bank_id = _make_bank(client, tmp_path)
    client.post(f'/api/video-bank/{bank_id}/probe', json={})
    client.post(f'/api/video-bank/{bank_id}/detect', json={})
    ids = client.get(f'/api/video-bank/{bank_id}/clips?ids_only=1').get_json()['ids']

    r = client.get(f'/api/video-bank/{bank_id}/clip/{ids[0]}/thumb')

    assert r.status_code == 404


# --- the target catalogue -------------------------------------------------------

def test_the_target_catalogue_is_served_with_its_caveats(client):
    """The frontend cannot hard-code these. Three fields decide whether a user
    wastes a week: `training_verified` (does the installed ai-toolkit have an
    architecture for it), `aitk_arch` (the string the training config needs, which
    is NOT our key — our wan22_ti2v5b is its wan22_5b), and `licence_note`.

    This test used to assert `wan22_ti2v5b.training_verified is False`, on the
    strength of web research about OTHER trainers. The installed ai-toolkit ships
    the architecture. Asserting a wrong fact is worse than asserting none: it
    defended the mistake."""
    body = client.get('/api/video/targets').get_json()

    by_key = {t['key']: t for t in body['targets']}
    assert by_key['wan22_14b']['fps'] == 16
    assert by_key['wan22_14b']['training_verified'] is True
    assert by_key['wan22_ti2v5b']['training_verified'] is True
    assert by_key['wan22_ti2v5b']['aitk_arch'] == 'wan22_5b'
    assert 'EU' in by_key['minimax_h3']['licence_note']
    assert 81 in by_key['wan22_14b']['frame_choices']


def test_each_target_says_how_long_its_default_clip_lasts(client):
    """"81 frames" means nothing to a user picking clips out of a rush; "5.0 s"
    does. Both Wan variants land on exactly 5.00 s at their own rate, which is the
    cross-check that the intervals arithmetic is right."""
    by_key = {t['key']: t for t in client.get('/api/video/targets')
              .get_json()['targets']}

    assert by_key['wan22_14b']['default_seconds'] == pytest.approx(5.0)
    assert by_key['wan22_ti2v5b']['default_seconds'] == pytest.approx(5.0)


# --- promotion ------------------------------------------------------------------

def test_promotion_answers_with_the_dataset_it_is_filling(client, tmp_path, seams):
    """202 and an id, so the UI can navigate straight to the dataset being built
    instead of guessing which one appeared."""
    bank_id = _ready_bank(client, tmp_path)

    r = client.post(f'/api/video-bank/{bank_id}/promote',
                    json={'name': 'wan set', 'target_profile': 'wan22_14b',
                          'frames': 81})

    assert r.status_code == 202, r.get_json()
    assert r.get_json()['id'] > 0


def test_a_frame_count_the_target_refuses_is_a_400_that_names_a_legal_one(
        client, tmp_path, seams):
    """29 is legal for Wan and illegal for LTX — the counter-example has to be
    picked with care, because every length Wan OFFERS also satisfies 8n+1."""
    bank_id = _ready_bank(client, tmp_path)

    r = client.post(f'/api/video-bank/{bank_id}/promote',
                    json={'name': 'x', 'target_profile': 'ltx23', 'frames': 29})

    assert r.status_code == 400
    assert '25' in r.get_json()['error']


def test_promoting_without_ffmpeg_is_a_503_before_anything_is_created(
        app, client, tmp_path, seams, monkeypatch):
    """503, not 400: nothing about the request is wrong, a tool is missing. And it
    has to land BEFORE the dataset row, or the user is left with an empty folder to
    clean up after a refusal."""
    from app.models import VideoDataset
    bank_id = _ready_bank(client, tmp_path)
    monkeypatch.setattr(svc, '_ffmpeg_or_raise', lambda: (_ for _ in ()).throw(
        RuntimeError('ffmpeg is required to cut clips and was not found')))

    r = client.post(f'/api/video-bank/{bank_id}/promote',
                    json={'name': 'x', 'target_profile': 'wan22_14b'})

    assert r.status_code == 503
    assert 'ffmpeg' in r.get_json()['error']
    with app.app_context():
        assert VideoDataset.query.count() == 0


def test_promotion_with_nothing_kept_is_a_400(client, tmp_path, seams):
    bank_id = _make_bank(client, tmp_path)
    client.post(f'/api/video-bank/{bank_id}/pipeline', json={})

    r = client.post(f'/api/video-bank/{bank_id}/promote',
                    json={'name': 'x', 'target_profile': 'wan22_14b'})

    assert r.status_code == 400


# --- the built dataset ----------------------------------------------------------

def _promote(client, tmp_path):
    bank_id = _ready_bank(client, tmp_path)
    r = client.post(f'/api/video-bank/{bank_id}/promote',
                    json={'name': 'wan set', 'target_profile': 'wan22_14b',
                          'frames': 81})
    assert r.status_code == 202, r.get_json()
    return bank_id, r.get_json()['id']


def test_the_dataset_payload_lists_its_clips_with_their_provenance(client, tmp_path,
                                                                   seams):
    _bank_id, ds_id = _promote(client, tmp_path)

    body = client.get(f'/api/video-dataset/{ds_id}').get_json()

    assert body['fps'] == 16 and body['frames'] == 81
    assert [i['filename'] for i in body['items']] == ['clip_0001.mp4',
                                                      'clip_0002.mp4']
    assert body['items'][1]['start_s'] == 41.25


def test_editing_a_caption_rewrites_the_sidecar_on_disk(client, tmp_path, seams):
    """THE test of this route. The trainer never reads our database — it reads the
    .txt next to the .mp4. A caption stored only in the DB trains the dataset on
    the previous text while the UI shows the new one, and nothing anywhere reveals
    it. So the write to disk is the feature; the row is the bookkeeping."""
    _bank_id, ds_id = _promote(client, tmp_path)
    body = client.get(f'/api/video-dataset/{ds_id}').get_json()
    item = body['items'][0]

    r = client.post(f'/api/video-dataset/{ds_id}/clip/{item["id"]}/caption',
                    json={'caption': 'a woman walking through a café at dusk'})

    assert r.status_code == 200
    sidecar = os.path.join(body['output_dir'], 'clip_0001.txt')
    assert open(sidecar, encoding='utf-8').read() == \
        'a woman walking through a café at dusk'


def test_clearing_a_caption_leaves_an_empty_file_never_a_missing_one(client,
                                                                     tmp_path, seams):
    """Deleting the sidecar would be the intuitive way to "remove" a caption and it
    is the one thing that must not happen: musubi-tuner raises FileNotFoundError
    out of a worker with no handler, and diffusion-pipe drops the clip."""
    _bank_id, ds_id = _promote(client, tmp_path)
    body = client.get(f'/api/video-dataset/{ds_id}').get_json()
    item = body['items'][0]
    client.post(f'/api/video-dataset/{ds_id}/clip/{item["id"]}/caption',
                json={'caption': 'something'})

    client.post(f'/api/video-dataset/{ds_id}/clip/{item["id"]}/caption',
                json={'caption': ''})

    assert os.path.isfile(os.path.join(body['output_dir'], 'clip_0001.txt'))


def test_deleting_a_dataset_is_a_404_when_it_is_not_yours(client):
    assert client.delete('/api/video-dataset/9999').status_code == 404


def test_deleting_a_dataset_removes_it_from_the_list(client, tmp_path, seams):
    _bank_id, ds_id = _promote(client, tmp_path)

    assert client.delete(f'/api/video-dataset/{ds_id}').status_code == 200

    assert client.get('/api/video-datasets').get_json()['datasets'] == []


# --- the metrics pass (wave 2) ---------------------------------------------------

def test_the_measure_pass_launches_and_reports_busy_like_every_other(
        client, tmp_path, seams, monkeypatch):
    """Same envelope as probe/detect/thumbs: 202 to launch. A user must not sense
    a seam between wave-1 passes and wave-2 ones."""
    from app.services import video_metrics_scan
    monkeypatch.setattr(video_metrics_scan, '_read_clip_frames',
                        lambda path, start, end, fps: [
                            {'luma': 0.5, 'sharp': 100.0, 'motion': 0.003}] * 10)
    bank_id = _ready_bank(client, tmp_path)

    r = client.post(f'/api/video-bank/{bank_id}/measure', json={})

    assert r.status_code == 202


def test_the_measure_pass_is_refused_while_the_extra_is_missing(
        client, tmp_path, seams, monkeypatch):
    """Measuring decodes, so without PyAV the pass cannot run — and the refusal
    names the missing piece rather than failing inside a worker."""
    bank_id = _ready_bank(client, tmp_path)
    from app import capabilities
    monkeypatch.setattr(capabilities, 'probe_video',
                        lambda: {'ok': False, 'decode': False, 'detect': False,
                                 'encode': True, 'detail': 'missing: av (video decoding)'})

    r = client.post(f'/api/video-bank/{bank_id}/measure', json={})

    assert r.status_code == 503
    assert 'av' in r.get_json()['error']


def test_clip_listings_carry_scores_and_flags(client, tmp_path, seams, app):
    """The grid filters on flags and sorts on scores, so both ride the clip rows.
    Flags are computed at read time — nothing verdict-like is stored."""
    bank_id = _ready_bank(client, tmp_path)
    import json as _json
    from app.extensions import db
    from app.models import VideoClip
    with app.app_context():
        clip = VideoClip.query.filter_by(bank_id=bank_id).first()
        clip.metrics_json = _json.dumps({
            'metrics_state': 'ok', 'motion_mean': 0.0001, 'motion_p95': 0.001,
            'luma_min': 0.5, 'luma_mean': 0.6, 'sharpness_p90': 200.0,
            'freeze_ratio': 0.0, 'sharpest_frame_s': 2.0})
        db.session.commit()

    # No cut is in force until the user chooses one — a default that filters
    # nothing is the design, since published thresholds measurably do not
    # transfer between corpora. So the flag appears only once a floor is set.
    from app import config
    config.save_config({'video_bank': {'motion_floor': 0.001}})

    body = client.get(f'/api/video-bank/{bank_id}/clips').get_json()

    row = body['clips'][0]
    assert row['metrics']['motion_mean'] == pytest.approx(0.0001)
    assert 'still' in row['flags']


def test_an_unmeasured_clip_lists_with_no_flags_and_no_metrics(client, tmp_path, seams):
    bank_id = _ready_bank(client, tmp_path)

    row = client.get(f'/api/video-bank/{bank_id}/clips').get_json()['clips'][0]

    assert row['metrics'] is None
    assert row['flags'] == []


def test_the_dry_run_endpoint_counts_per_rule_before_anything_is_cut(
        client, tmp_path, seams, app):
    """The mode that keeps a mis-set threshold from quietly keeping 3% of a bank:
    per-rule counts, computed against the bank's stored raw scores."""
    bank_id = _ready_bank(client, tmp_path)
    import json as _json
    from app.extensions import db
    from app.models import VideoClip
    with app.app_context():
        clip = VideoClip.query.filter_by(bank_id=bank_id).first()
        clip.metrics_json = _json.dumps({
            'metrics_state': 'ok', 'motion_mean': 0.0001, 'motion_p95': 0.001,
            'luma_min': 0.5, 'luma_mean': 0.6, 'sharpness_p90': 200.0,
            'freeze_ratio': 0.0, 'sharpest_frame_s': 2.0})
        db.session.commit()

    r = client.post(f'/api/video-bank/{bank_id}/metrics-dry-run',
                    json={'motion_floor': 0.001})

    body = r.get_json()
    assert body['still'] == 1
    assert body['total_flagged'] == 1


# --- retouching the cuts: bounds, split, hand-made shots -------------------------
#
# Until these three routes existed, a detector that missed a boundary — or a shot
# holding a frozen tail — cost the WHOLE shot: the only available gesture was
# ✕ Reject. The schema already said `detector='manual'` and nothing could write it.
#
# Every test here is really about the same invariant, stated three ways: a
# thumbnail and a set of measurements belong to the bounds they were taken on, and
# the moment those bounds move they become claims about a shot that no longer
# exists. They are forgotten rather than kept and clamped.

def _cut_bank(client, tmp_path):
    """A detected bank whose first shot is 0 → 8 s and second 41.25 → 50 s — the
    state a user retouches from. The source is 120 s long (see `seams`)."""
    bank_id = _make_bank(client, tmp_path)
    assert client.post(f'/api/video-bank/{bank_id}/pipeline',
                       json={}).status_code == 202
    return bank_id


def _clips(client, bank_id):
    return client.get(f'/api/video-bank/{bank_id}/clips').get_json()['clips']


def test_adjusting_the_bounds_moves_them_and_calls_the_cut_manual(client, tmp_path,
                                                                  seams):
    """`detector` is persisted per clip precisely so "why is this cut here?" stays
    answerable on a bank worked on over days. Once a human has moved a boundary the
    answer is no longer 'transnetv2', and leaving that string would credit the
    detector for a decision it did not make."""
    bank_id = _cut_bank(client, tmp_path)
    clip = _clips(client, bank_id)[0]

    r = client.patch(f'/api/video-bank/{bank_id}/clip/{clip["id"]}/bounds',
                     json={'start_s': 1.5, 'end_s': 6.25})

    assert r.status_code == 200, r.get_json()
    row = r.get_json()['clip']
    assert (row['start_s'], row['end_s']) == (1.5, 6.25)
    assert row['detector'] == 'manual'


def test_adjusting_the_bounds_forgets_the_thumbnail_and_the_measurements(
        client, tmp_path, seams, app):
    """The point of the whole feature. A thumbnail taken at 4 s of a shot that now
    starts at 6 s is not stale, it is WRONG — it shows a frame the shot no longer
    contains. Same for the metrics: motion and freeze ratio were integrated over a
    span that has changed. The thumbs pass clamps a periodic timestamp, which
    protects against a crash, not against a lie; forgetting is the honest answer,
    and it also makes `counts.thumbs` drop so the workspace's next-step line asks
    for the thumbnails again on its own."""
    import json as _json
    from app.extensions import db
    from app.models import VideoClip
    bank_id = _cut_bank(client, tmp_path)
    clip = _clips(client, bank_id)[0]
    assert clip['thumb_state'] == 'ok'
    # The seam does not write bytes; the file is what the grid actually reads, so
    # this test puts a real one there to watch it go.
    thumb = svc.thumb_path(bank_id, clip['id'])
    thumb.parent.mkdir(parents=True, exist_ok=True)
    thumb.write_bytes(b'\xff\xd8\xff')
    with app.app_context():
        row = db.session.get(VideoClip, clip['id'])
        row.metrics_json = _json.dumps({'metrics_state': 'ok', 'motion_mean': 0.5,
                                        'sharpest_frame_s': 4.0})
        db.session.commit()

    body = client.patch(f'/api/video-bank/{bank_id}/clip/{clip["id"]}/bounds',
                        json={'start_s': 6.0, 'end_s': 8.0}).get_json()

    assert body['clip']['thumb_state'] is None
    assert body['clip']['metrics'] is None
    # The FILE goes too: the grid reads the thumb URL, not the column, so a leftover
    # JPEG would keep showing the old frame until the pass ran again.
    assert not thumb.is_file()
    assert body['counts']['thumbs'] == 1


def test_bounds_beyond_the_end_of_the_source_are_refused(client, tmp_path, seams):
    """The source is 120 s. `ffmpeg -ss` past the end does not fail — it produces a
    zero-length or one-frame file at promotion, hours after the mistake."""
    bank_id = _cut_bank(client, tmp_path)
    clip = _clips(client, bank_id)[0]

    r = client.patch(f'/api/video-bank/{bank_id}/clip/{clip["id"]}/bounds',
                     json={'start_s': 100.0, 'end_s': 130.0})

    assert r.status_code == 400
    assert '120' in r.get_json()['error']


@pytest.mark.parametrize('start_s,end_s', [
    (5.0, 5.0),        # empty
    (6.0, 5.0),        # inverted
    (5.0, 5.2),        # 0.2 s — below the floor any target could ingest
    (-1.0, 4.0),       # before the file starts
])
def test_a_range_that_cannot_be_trained_on_is_refused(client, tmp_path, seams,
                                                      start_s, end_s):
    """An inverted range is also what `clipFragmentSrc` refuses on the client, and
    for the same reason: a malformed media fragment does not throw, the browser
    ignores it and plays the whole two-hour rush."""
    bank_id = _cut_bank(client, tmp_path)
    clip = _clips(client, bank_id)[0]

    r = client.patch(f'/api/video-bank/{bank_id}/clip/{clip["id"]}/bounds',
                     json={'start_s': start_s, 'end_s': end_s})

    assert r.status_code == 400, r.get_json()


def test_a_promoted_clip_stays_editable_and_the_dataset_keeps_its_own_bounds(
        client, tmp_path, seams):
    """Provenance is a SNAPSHOT: VideoDatasetClip copies relpath and bounds at
    encode time, so re-cutting the bank clip afterwards cannot retro-edit what was
    already encoded. Refusing the edit would therefore protect nothing and would
    make the second, better cut impossible — which is exactly when someone wants
    it, having just watched the built dataset."""
    bank_id, ds_id = _promote(client, tmp_path)
    clip = [c for c in _clips(client, bank_id) if c['start_s'] == 41.25][0]

    r = client.patch(f'/api/video-bank/{bank_id}/clip/{clip["id"]}/bounds',
                     json={'start_s': 42.0, 'end_s': 47.0})

    assert r.status_code == 200, r.get_json()
    items = client.get(f'/api/video-dataset/{ds_id}').get_json()['items']
    assert items[1]['start_s'] == 41.25


def test_splitting_a_shot_makes_two_and_the_new_half_inherits_the_status(
        client, tmp_path, seams):
    """Falling back to `pending` on both halves would undo the decision the user was
    in the middle of making: you split a KEPT shot because its tail is bad, and the
    half you are keeping must not silently leave the keep pile."""
    bank_id = _cut_bank(client, tmp_path)
    clip = _clips(client, bank_id)[0]
    client.post(f'/api/video-bank/{bank_id}/triage',
                json={'ids': [clip['id']], 'status': 'keep'})

    r = client.post(f'/api/video-bank/{bank_id}/clip/{clip["id"]}/split',
                    json={'at_s': 5.0})

    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert (body['clip']['start_s'], body['clip']['end_s']) == (0.0, 5.0)
    assert (body['new_clip']['start_s'], body['new_clip']['end_s']) == (5.0, 8.0)
    assert body['new_clip']['status'] == 'keep'
    assert body['new_clip']['detector'] == body['clip']['detector'] == 'manual'
    assert body['counts']['clips'] == 3


def test_a_split_forgets_the_thumbnail_of_BOTH_halves(client, tmp_path, seams):
    """The parent keeps its start, so it is tempting to let it keep its thumbnail —
    but the frame was taken from the MIDDLE of the old span, which is now inside the
    other half."""
    bank_id = _cut_bank(client, tmp_path)
    clip = _clips(client, bank_id)[0]

    body = client.post(f'/api/video-bank/{bank_id}/clip/{clip["id"]}/split',
                       json={'at_s': 5.0}).get_json()

    assert body['clip']['thumb_state'] is None
    assert body['new_clip']['thumb_state'] is None


@pytest.mark.parametrize('at_s', [0.0, 8.0, 9.0, 0.2, 7.9])
def test_a_split_point_outside_or_flush_against_a_bound_is_refused(
        client, tmp_path, seams, at_s):
    """A split at the boundary makes an empty shot; a split 0.2 s in makes one no
    target can ingest. Both are 400 rather than a silently clamped cut, because the
    user is looking at a playhead and would not see the clamp."""
    bank_id = _cut_bank(client, tmp_path)
    clip = _clips(client, bank_id)[0]

    r = client.post(f'/api/video-bank/{bank_id}/clip/{clip["id"]}/split',
                    json={'at_s': at_s})

    assert r.status_code == 400, r.get_json()


def test_a_shot_the_detector_missed_can_be_cut_by_hand(client, tmp_path, seams):
    """The other half of the same problem: a boundary the detector did not draw at
    all. A hand-made shot is `pending` because it has never been judged, and it
    lands in the gallery ordered by start like any other."""
    bank_id = _cut_bank(client, tmp_path)
    source_id = _clips(client, bank_id)[0]['source_id']

    r = client.post(f'/api/video-bank/{bank_id}/source/{source_id}/clips',
                    json={'start_s': 20.0, 'end_s': 24.0})

    assert r.status_code == 200, r.get_json()
    row = r.get_json()['clip']
    assert row['detector'] == 'manual' and row['status'] == 'pending'
    assert row['thumb_state'] is None
    assert [c['start_s'] for c in _clips(client, bank_id)] == [0.0, 20.0, 41.25]


def test_a_hand_made_shot_on_an_unknown_source_is_a_404(client, tmp_path, seams):
    bank_id = _cut_bank(client, tmp_path)

    r = client.post(f'/api/video-bank/{bank_id}/source/9999/clips',
                    json={'start_s': 1.0, 'end_s': 4.0})

    assert r.status_code == 404


def test_a_clip_that_belongs_to_another_bank_is_a_404(client, tmp_path, seams):
    """The bank id in the path is not decoration. Without the pairing check, a clip
    id from bank A would be editable through bank B's URL — the same shape of hole
    the media route closes by resolving the source THROUGH its bank."""
    bank_a = _cut_bank(client, tmp_path)
    bank_b = _make_bank(client, tmp_path / 'other')
    clip = _clips(client, bank_a)[0]

    r = client.patch(f'/api/video-bank/{bank_b}/clip/{clip["id"]}/bounds',
                     json={'start_s': 1.0, 'end_s': 4.0})

    assert r.status_code == 404


@pytest.mark.parametrize('verb,path,body', [
    ('patch', 'clip/{clip}/bounds', {'start_s': 1.0, 'end_s': 4.0}),
    ('post', 'clip/{clip}/split', {'at_s': 4.0}),
    ('post', 'source/{source}/clips', {'start_s': 20.0, 'end_s': 24.0}),
])
def test_retouching_is_refused_while_a_pass_owns_the_bank(client, tmp_path, seams,
                                                          verb, path, body):
    """409 with `busy_kind`, exactly like a second pass. Not pedantry: the thumbs
    pass reads bounds and then stamps `thumb_state='ok'`, so an edit landing between
    those two writes produces a thumbnail of the OLD span marked as current — the
    one failure mode this whole feature exists to prevent, made invisible."""
    import time
    from app.services import bank_jobs
    bank_id = _cut_bank(client, tmp_path)
    clip = _clips(client, bank_id)[0]
    bank_jobs._jobs[svc.job_key(bank_id)] = {
        'kind': 'thumbs', 'done': 1, 'total': 9, 'error': None, 'cancelled': False,
        'finished': False, 'detail': None, 'started_at': time.time(),
        '_touched': time.time(), '_cancel_hook': None, 'pipeline': None}

    url = (f'/api/video-bank/{bank_id}/'
           + path.format(clip=clip['id'], source=clip['source_id']))
    r = getattr(client, verb)(url, json=body)

    assert r.status_code == 409, r.get_json()
    assert r.get_json()['busy_kind'] == 'thumbs'


def test_re_detecting_a_file_never_destroys_a_hand_made_cut(client, tmp_path, seams):
    """A re-detect deletes the clips of the file it re-cuts, sparing only the ones
    already promoted. Manual cuts were not spared by that rule and would have been
    wiped by a checkbox — the single most expensive way to lose an afternoon of
    triage. Detector-drawn clips still go, which is the point of re-detecting."""
    bank_id = _cut_bank(client, tmp_path)
    source_id = _clips(client, bank_id)[0]['source_id']
    client.post(f'/api/video-bank/{bank_id}/source/{source_id}/clips',
                json={'start_s': 20.0, 'end_s': 24.0})

    assert client.post(f'/api/video-bank/{bank_id}/detect',
                       json={'redetect': True}).status_code == 202

    rows = _clips(client, bank_id)
    assert [c for c in rows if c['detector'] == 'manual']
    assert len([c for c in rows if c['detector'] == 'transnetv2']) == 2
