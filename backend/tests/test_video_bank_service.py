"""🎬 The video bank service — bounds in, one flat dataset out.

The architectural promise of this lane is that a bank is a set of DECISIONS, not
a pile of media. It stores where each shot begins and ends and whether you want
it; the only bytes it ever writes are thumbnails. Encoding happens once, at
promotion, on the clips that survived triage. That is what keeps a 340-clip bank
from costing 340 encodes to keep 128 of them, and it is what keeps the user's
source folder readable-only in the literal sense.

Every test here defends one of the ways that promise, or the trainers reading its
output, could be broken quietly:

  * a source folder whose files are named .MOV rather than .mov scans as EMPTY —
    the bank is created, the user is told nothing, and the folder looks broken;
  * a rejected clip that still reaches ffmpeg costs minutes per bank and defeats
    the whole design;
  * a clip cut from frame INDICES drifts on variable-frame-rate material, which
    is most scraped material;
  * a missing .txt sidecar CRASHES musubi-tuner and is silently DROPPED by
    diffusion-pipe;
  * a subfolder under a dataset root is picked up by ai-toolkit's RECURSIVE scan
    and trained on without a word.

No ffmpeg, no PyAV, no torch: probing, detection, thumbnailing and encoding are
each a single monkeypatchable seam, because CI has none of them installed.
"""
import os

import pytest

from app.config import LOCAL_USER
from app.extensions import db
from app.models import VideoClip, VideoDataset, VideoDatasetClip, VideoSource
from app.services import video_bank_service as svc

from _video_extra import detect_source_stub


# --- fakes for the four media seams -------------------------------------------

def _fake_probe(_path):
    """A readable 30 fps 1080p file. The shape services/video_probe.probe returns."""
    return {'duration_s': 120.0, 'fps_native': 30.0, 'width': 1920, 'height': 1080,
            'codec': 'h264', 'probe_state': 'ok', 'file_size': 4096}


def _fake_shots(_path, _fps_native=None):
    """Two shots, in PTS seconds. Long enough for an 81-frame clip at 16 fps (5 s)."""
    return [{'start_s': 0.0, 'end_s': 8.0, 'start_frame': 0, 'end_frame': 240},
            {'start_s': 41.25, 'end_s': 50.0, 'start_frame': 1237, 'end_frame': 1500}]


@pytest.fixture()
def seams(monkeypatch):
    """Install the four media seams and record every ffmpeg invocation."""
    calls = []

    def _run(args):
        calls.append(list(args))
        # ffmpeg's last argument is the destination; make it exist so the caller's
        # bookkeeping sees the same world a real encode would leave behind.
        with open(args[-1], 'wb') as fh:
            fh.write(b'\x00')
        return 0, ''

    monkeypatch.setattr(svc, '_probe_file', _fake_probe)
    monkeypatch.setattr(svc, '_detect_source', detect_source_stub(_fake_shots(None)))
    monkeypatch.setattr(svc, '_write_thumbnail', lambda *a, **k: True)
    monkeypatch.setattr(svc, '_run_ffmpeg', _run)
    monkeypatch.setattr(svc, '_ffmpeg_or_raise', lambda: '/usr/bin/ffmpeg')
    return calls


def _source_folder(tmp_path, names=('a.mp4', 'b.MOV')):
    folder = tmp_path / 'rushes'
    folder.mkdir(parents=True, exist_ok=True)
    for n in names:
        (folder / n).write_bytes(b'\x00' * 32)
    return str(folder)


def _bank(app, tmp_path, names=('a.mp4', 'b.MOV')):
    bank, added = svc.create_bank(LOCAL_USER, 'rushes', _source_folder(tmp_path, names))
    return bank.id, added


# --- what counts as a source video --------------------------------------------

def test_the_extension_match_is_case_insensitive(app, tmp_path):
    """`DSC_0001.MOV` is what a camera writes and what half of scraped material
    carries. Matching extensions without folding the case creates the bank, reports
    zero files and says nothing — the folder simply looks empty, which reads as a
    broken app rather than a naming detail."""
    with app.app_context():
        _bank_id, added = _bank(app, tmp_path, ('a.mp4', 'B.MOV', 'c.MKV',
                                                'd.WebM', 'e.AVI'))

        assert added == 5


def test_non_video_files_in_the_same_folder_are_ignored(app, tmp_path):
    """Rush folders hold thumbnails, .srt subtitles and stray .txt notes. A bank
    that inventories them would hand unopenable files to the probe and report a
    wall of unreadable sources."""
    with app.app_context():
        _bank_id, added = _bank(app, tmp_path,
                                ('a.mp4', 'cover.jpg', 'notes.txt', 'subs.srt'))

        assert added == 1


def test_a_rescan_adds_only_what_appeared(app, tmp_path):
    """A bank points at a LIVE folder. The re-walk must be strictly additive: a
    triage worked over days cannot be reset because someone dropped one more file
    into the folder."""
    with app.app_context():
        bank_id, _ = _bank(app, tmp_path, ('a.mp4',))
        folder = svc.get_bank(LOCAL_USER, bank_id).source_path
        (open(os.path.join(folder, 'later.mp4'), 'wb')).write(b'\x00')

        sync = svc.refresh_bank(LOCAL_USER, bank_id, force=True)

        assert sync['added'] == 1
        assert VideoSource.query.filter_by(bank_id=bank_id).count() == 2


def test_a_vanished_file_is_counted_never_deleted(app, tmp_path):
    """An unplugged drive must not wipe a triage. The image lane learned this the
    expensive way; the video lane inherits the rule rather than the incident."""
    with app.app_context():
        bank_id, _ = _bank(app, tmp_path, ('a.mp4', 'b.mp4'))
        folder = svc.get_bank(LOCAL_USER, bank_id).source_path
        os.remove(os.path.join(folder, 'b.mp4'))

        sync = svc.refresh_bank(LOCAL_USER, bank_id, force=True)

        assert sync['missing'] == 1
        assert VideoSource.query.filter_by(bank_id=bank_id).count() == 2


# --- probing and detection are per FILE ---------------------------------------

def test_probing_writes_the_per_file_facts(app, tmp_path, seams):
    """Duration, native rate and geometry are properties of the FILE. Storing them
    per clip would re-probe the same two-hour rush four hundred times."""
    with app.app_context():
        bank_id, _ = _bank(app, tmp_path, ('a.mp4',))

        svc.start_probe(app, LOCAL_USER, bank_id)

        src = VideoSource.query.filter_by(bank_id=bank_id).one()
        assert src.probe_state == 'ok'
        assert src.fps_native == 30.0
        assert (src.width, src.height) == (1920, 1080)


def test_one_unreadable_file_does_not_sink_the_probe_pass(app, tmp_path, monkeypatch,
                                                          seams):
    """Four hundred files, one truncated download. The bank must cost that file and
    not the pass — a probe that raises leaves every later file unprobed and the
    user with no way to tell which one was bad."""
    def _probe(path):
        if path.endswith('bad.mp4'):
            return {'duration_s': None, 'fps_native': None, 'width': None,
                    'height': None, 'codec': None, 'probe_state': 'unreadable',
                    'file_size': 0}
        return _fake_probe(path)

    monkeypatch.setattr(svc, '_probe_file', _probe)
    with app.app_context():
        bank_id, _ = _bank(app, tmp_path, ('bad.mp4', 'good.mp4'))

        svc.start_probe(app, LOCAL_USER, bank_id)

        states = {s.relpath: s.probe_state
                  for s in VideoSource.query.filter_by(bank_id=bank_id)}
        assert states == {'bad.mp4': 'unreadable', 'good.mp4': 'ok'}


def test_detection_stores_the_bounds_in_pts_seconds(app, tmp_path, seams):
    """start_s/end_s are the canonical form. Frame indices are kept because they
    are what the detector said, but scraped material is routinely variable-frame-
    rate, where index n names no stable instant."""
    with app.app_context():
        bank_id, _ = _bank(app, tmp_path, ('a.mp4',))
        svc.start_probe(app, LOCAL_USER, bank_id)

        svc.start_detect(app, LOCAL_USER, bank_id)

        clips = VideoClip.query.filter_by(bank_id=bank_id).order_by(
            VideoClip.start_s).all()
        assert [c.start_s for c in clips] == [0.0, 41.25]
        assert clips[1].start_frame == 1237       # kept, and merely informative


def test_a_detector_failure_is_recorded_on_the_file_and_the_pass_continues(
        app, tmp_path, monkeypatch, seams):
    """detect_state is separate from probe_state because a file can be perfectly
    readable and still defeat the detector. Collapsing the two would send the user
    to reinstall the decoder for a detection problem."""
    def _detect(path, _fps=None, **_options):
        if path.endswith('bad.mp4'):
            raise RuntimeError('detector exploded')
        return detect_source_stub(_fake_shots(path))(path)

    monkeypatch.setattr(svc, '_detect_source', _detect)
    with app.app_context():
        bank_id, _ = _bank(app, tmp_path, ('bad.mp4', 'good.mp4'))
        svc.start_probe(app, LOCAL_USER, bank_id)

        svc.start_detect(app, LOCAL_USER, bank_id)

        states = {s.relpath: s.detect_state
                  for s in VideoSource.query.filter_by(bank_id=bank_id)}
        assert states == {'bad.mp4': 'error', 'good.mp4': 'ok'}
        assert VideoClip.query.filter_by(bank_id=bank_id).count() == 2


def test_a_missing_detection_extra_fails_the_pass_not_every_file(
        app, tmp_path, monkeypatch, seams):
    """`ShotDetectUnavailable` is a fact about the INSTALL; `ShotDetectFileError` is
    a fact about one file. Folding the first into the second stamps detect_state=
    'error' onto all four hundred sources for a missing pip package — and because
    the pass then skips anything already marked, installing the extra afterwards
    fixes nothing until the user finds the re-detect checkbox. So the pass stops,
    says why once, and leaves every file untouched and re-detectable.

    Recognised by CLASS NAME rather than isinstance: the whole point is that the
    module may be absent, and importing it inside the handler to identify an error
    raised by its absence is circular."""
    class ShotDetectUnavailable(RuntimeError):
        pass

    def _detect(_path, _fps=None, **_options):
        raise ShotDetectUnavailable('transnetv2-pytorch is not installed')

    monkeypatch.setattr(svc, '_detect_source', _detect)
    with app.app_context():
        bank_id, _ = _bank(app, tmp_path, ('a.mp4', 'b.mp4'))
        svc.start_probe(app, LOCAL_USER, bank_id)

        job = svc.start_detect(app, LOCAL_USER, bank_id)

        assert 'not installed' in (job['error'] or '')
        assert [s.detect_state for s in
                VideoSource.query.filter_by(bank_id=bank_id)] == [None, None]


def test_the_source_folder_is_never_written_to(app, tmp_path, seams):
    """The bank's whole contract. Thumbnails go to the app's own working area; the
    user's rushes folder must come out of a full pass byte-identical in listing."""
    with app.app_context():
        bank_id, _ = _bank(app, tmp_path, ('a.mp4',))
        folder = svc.get_bank(LOCAL_USER, bank_id).source_path
        before = sorted(os.listdir(folder))

        svc.start_pipeline(app, LOCAL_USER, bank_id)

        assert sorted(os.listdir(folder)) == before


# --- triage --------------------------------------------------------------------

def test_triage_records_keep_and_reject(app, tmp_path, seams):
    with app.app_context():
        bank_id, _ = _bank(app, tmp_path, ('a.mp4',))
        svc.start_pipeline(app, LOCAL_USER, bank_id)
        ids = [c.id for c in VideoClip.query.filter_by(bank_id=bank_id)
               .order_by(VideoClip.id)]

        svc.set_clip_status(LOCAL_USER, bank_id, [ids[0]], 'keep')
        svc.set_clip_status(LOCAL_USER, bank_id, [ids[1]], 'reject', reason='blurry')

        rows = {c.id: (c.status, c.reject_reason)
                for c in VideoClip.query.filter_by(bank_id=bank_id)}
        assert rows[ids[0]] == ('keep', None)
        assert rows[ids[1]] == ('reject', 'blurry')


# --- promotion: the only place media is written -------------------------------

def _promoted(app, tmp_path, seams, *, profile='wan22_14b', frames=81, **kw):
    """A bank taken all the way: scanned, detected, both clips kept, promoted."""
    bank_id, _ = _bank(app, tmp_path, ('a.mp4',))
    svc.start_pipeline(app, LOCAL_USER, bank_id)
    ids = [c.id for c in VideoClip.query.filter_by(bank_id=bank_id)
           .order_by(VideoClip.id)]
    svc.set_clip_status(LOCAL_USER, bank_id, ids, 'keep')
    out = svc.start_promote(app, LOCAL_USER, bank_id, ids=None, name='set',
                            target_profile=profile, frames=frames, **kw)
    return bank_id, ids, out


def test_only_kept_clips_are_encoded(app, tmp_path, seams):
    """THE architectural decision of this lane. Encoding at detection time would
    cost 340 encodes to keep 128, and would write media the user never asked for
    into a bank whose contract says it stores bounds."""
    with app.app_context():
        bank_id, _ = _bank(app, tmp_path, ('a.mp4',))
        svc.start_pipeline(app, LOCAL_USER, bank_id)
        ids = [c.id for c in VideoClip.query.filter_by(bank_id=bank_id)
               .order_by(VideoClip.id)]
        assert not seams, 'detection must not have encoded anything'
        svc.set_clip_status(LOCAL_USER, bank_id, [ids[0]], 'keep')
        svc.set_clip_status(LOCAL_USER, bank_id, [ids[1]], 'reject')

        svc.start_promote(app, LOCAL_USER, bank_id, ids=None, name='set',
                          target_profile='wan22_14b', frames=81)

        assert len(seams) == 1


def test_the_cut_starts_at_the_pts_bound_not_a_frame_index(app, tmp_path, seams):
    """`ffmpeg -ss` takes seconds. Reconstructing a timestamp from the detector's
    frame index and the source's average rate drifts on VFR material — which is
    most scraped material — and the drift is invisible in the output."""
    with app.app_context():
        _promoted(app, tmp_path, seams)

        starts = {args[args.index('-ss') + 1] for args in seams}
        assert '41.250000' in starts


def test_every_encoded_clip_gets_a_caption_sidecar_even_when_empty(app, tmp_path,
                                                                   seams):
    """Wave 1 produces no captions at all, so without this EVERY clip lands on the
    missing-sidecar path: musubi-tuner raises FileNotFoundError out of a worker
    future with no handler, and diffusion-pipe drops the clip silently because its
    skip_empty_caption defaults to true."""
    with app.app_context():
        _bank_id, _ids, out = _promoted(app, tmp_path, seams)
        out_dir = db.session.get(VideoDataset, out['id']).output_dir

        names = sorted(os.listdir(out_dir))
        assert names == ['clip_0001.mp4', 'clip_0001.txt',
                         'clip_0002.mp4', 'clip_0002.txt']
        assert open(os.path.join(out_dir, 'clip_0001.txt'),
                    encoding='utf-8').read() == ''


def test_the_dataset_folder_is_flat(app, tmp_path, seams):
    """ai-toolkit's dataset scan is os.walk — RECURSIVE — and excludes only
    dotfiles and a folder literally named `_controls`. Any subfolder we wrote for
    our own convenience (previews, rejects, per-bucket splits) would be trained on
    silently. So the rule is not "we prefer flat", it is "a subfolder is a defect"."""
    with app.app_context():
        _bank_id, _ids, out = _promoted(app, tmp_path, seams)
        out_dir = db.session.get(VideoDataset, out['id']).output_dir

        subdirs = [d for _root, dirs, _files in os.walk(out_dir) for d in dirs]
        assert subdirs == []


def test_a_clip_too_short_for_the_target_is_skipped_and_counted(app, tmp_path,
                                                                monkeypatch, seams):
    """Handed 2 s where 5 are needed, ffmpeg writes a 32-frame file and exits 0.
    ai-toolkit then clamps and repeats frames, training it as stills, and nothing
    anywhere says so. Skipping loudly is the only honest answer — and the clip must
    leave NO half-file and no orphan sidecar behind."""
    monkeypatch.setattr(svc, '_detect_source', detect_source_stub([
        {'start_s': 0.0, 'end_s': 8.0, 'start_frame': 0, 'end_frame': 240},
        {'start_s': 20.0, 'end_s': 22.0, 'start_frame': 600, 'end_frame': 660},
    ]))
    with app.app_context():
        _bank_id, _ids, out = _promoted(app, tmp_path, seams)
        ds = db.session.get(VideoDataset, out['id'])

        assert sorted(os.listdir(ds.output_dir)) == ['clip_0001.mp4', 'clip_0001.txt']
        assert VideoDatasetClip.query.filter_by(dataset_id=ds.id).count() == 1


def test_the_dataset_remembers_what_it_was_encoded_at(app, tmp_path, seams):
    """fps and frames are denormalised on purpose: the files on disk are already
    committed to them, so a dataset built last month must keep reporting what it
    actually contains even if the profile's defaults move."""
    with app.app_context():
        _bank_id, _ids, out = _promoted(app, tmp_path, seams)
        ds = db.session.get(VideoDataset, out['id'])

        assert (ds.target_profile, ds.fps, ds.frames) == ('wan22_14b', 16, 81)


def test_each_dataset_clip_keeps_the_bounds_it_was_cut_at(app, tmp_path, seams):
    """Provenance is what makes a re-export to another target a re-encode rather
    than a re-scan. It must outlive the bank, which is a scratch container the user
    is free to delete — hence plain integers, not foreign keys."""
    with app.app_context():
        _bank_id, _ids, out = _promoted(app, tmp_path, seams)

        row = VideoDatasetClip.query.filter_by(
            dataset_id=out['id'], filename='clip_0002.mp4').one()
        assert (row.start_s, row.src_relpath) == (41.25, 'a.mp4')


def test_promoted_clips_are_marked_and_rejected_ones_are_not(app, tmp_path, seams):
    with app.app_context():
        bank_id, _ids, out = _promoted(app, tmp_path, seams)

        marked = VideoClip.query.filter_by(
            bank_id=bank_id, promoted_dataset_id=out['id']).count()
        assert marked == 2


def test_a_frame_count_the_target_cannot_ingest_is_refused_by_name(app, tmp_path,
                                                                   seams):
    """LTX compresses time by 8, so 29 frames is legal for Wan and illegal for LTX.
    Nothing downstream objects: diffusion-pipe rounds the bucket down in silence.
    Refusing here is the only place the rule can be enforced, and the refusal names
    the nearest count that works so it is actionable rather than a wall.

    29 and not 33 — the counter-example has to be picked with care, because every
    length Wan actually OFFERS also satisfies 8n+1. (video_targets' own module
    docstring names 33 here, which is 8*4+1 and therefore perfectly legal for LTX;
    the rule is right, that one sentence is not. test_video_targets already uses
    29.)"""
    with app.app_context():
        bank_id, _ = _bank(app, tmp_path, ('a.mp4',))
        svc.start_pipeline(app, LOCAL_USER, bank_id)
        svc.set_clip_status(LOCAL_USER, bank_id, [], 'keep')

        with pytest.raises(ValueError) as err:
            svc.start_promote(app, LOCAL_USER, bank_id, ids=None, name='set',
                              target_profile='ltx23', frames=29)

        assert '25' in str(err.value)


def test_a_resolution_off_the_targets_grid_is_refused(app, tmp_path, seams):
    """A step, not a whitelist: Wan's 5B variant patches 2x2 over a VAE that
    compresses space by 16, which is why its official 720p size is 1280x704. A
    dataset built at 1280x720 for it is silently resized by the trainer."""
    with app.app_context():
        bank_id, _ = _bank(app, tmp_path, ('a.mp4',))
        svc.start_pipeline(app, LOCAL_USER, bank_id)
        svc.set_clip_status(LOCAL_USER, bank_id, [], 'keep')

        with pytest.raises(ValueError):
            svc.start_promote(app, LOCAL_USER, bank_id, ids=None, name='set',
                              target_profile='wan22_ti2v5b', frames=121,
                              size=(1280, 720))


def test_promotion_without_a_single_kept_clip_is_refused(app, tmp_path, seams):
    """An empty dataset folder is a worse outcome than a refusal: the trainer
    fails much later, with a message about the dataset config."""
    with app.app_context():
        bank_id, _ = _bank(app, tmp_path, ('a.mp4',))
        svc.start_pipeline(app, LOCAL_USER, bank_id)

        with pytest.raises(ValueError):
            svc.start_promote(app, LOCAL_USER, bank_id, ids=None, name='set',
                              target_profile='wan22_14b', frames=81)


# --- the job slot --------------------------------------------------------------

def test_a_video_bank_does_not_occupy_the_image_bank_of_the_same_id(app, tmp_path,
                                                                    seams):
    """bank_jobs keys its registry by bank id, and the two lanes number their banks
    independently — so image bank 1 and video bank 1 both exist and are different
    things. Sharing the raw key makes a video detection pass refuse a click on an
    unrelated image bank, which is unexplainable from the UI."""
    from app.services import bank_jobs
    with app.app_context():
        bank_id, _ = _bank(app, tmp_path, ('a.mp4',))
        bank_jobs._jobs[bank_id] = {
            'kind': 'score', 'done': 0, 'total': 1, 'error': None,
            'cancelled': False, 'finished': False, 'detail': None,
            'started_at': 0.0, '_touched': __import__('time').time(),
            '_cancel_hook': None, 'pipeline': None}

        svc.start_probe(app, LOCAL_USER, bank_id)      # must NOT raise BankJobBusy

        assert VideoSource.query.filter_by(bank_id=bank_id).one().probe_state == 'ok'


def test_a_second_pass_on_a_busy_video_bank_is_refused(app, tmp_path, seams):
    """One live pass per bank, same as the image lane — two detections writing
    clips for the same file would double every shot."""
    from app.services import bank_jobs
    with app.app_context():
        bank_id, _ = _bank(app, tmp_path, ('a.mp4',))
        bank_jobs._jobs[svc.job_key(bank_id)] = {
            'kind': 'detect', 'done': 0, 'total': 1, 'error': None,
            'cancelled': False, 'finished': False, 'detail': None,
            'started_at': 0.0, '_touched': __import__('time').time(),
            '_cancel_hook': None, 'pipeline': None}

        with pytest.raises(bank_jobs.BankJobBusy):
            svc.start_probe(app, LOCAL_USER, bank_id)


def test_the_pipeline_outcome_survives_the_night(app, tmp_path, seams):
    """A pass launched at midnight is read the next morning. The in-memory job
    registry dies with the process, so the report is persisted on the bank — same
    role as ImageBank.pipeline_report."""
    with app.app_context():
        bank_id, _ = _bank(app, tmp_path, ('a.mp4',))

        svc.start_pipeline(app, LOCAL_USER, bank_id)

        report = svc.bank_payload(LOCAL_USER, bank_id)['pipeline_report']
        assert [s['step'] for s in report['steps']] == list(svc.PIPELINE_STEPS)
        assert all(s['status'] == 'done' for s in report['steps'])


# --- deletion ------------------------------------------------------------------

def test_deleting_a_bank_takes_its_sources_and_clips(app, tmp_path, seams):
    """A bank is a scratch container. What it must NOT take is a dataset built out
    of it — that is why VideoDatasetClip's provenance is not a foreign key."""
    with app.app_context():
        bank_id, _ids, out = _promoted(app, tmp_path, seams)

        assert svc.delete_bank(LOCAL_USER, bank_id) is True

        assert VideoSource.query.filter_by(bank_id=bank_id).count() == 0
        assert VideoClip.query.filter_by(bank_id=bank_id).count() == 0
        assert VideoDatasetClip.query.filter_by(dataset_id=out['id']).count() == 2


def test_deleting_a_dataset_leaves_the_banks_triage_alone(app, tmp_path, seams):
    """Throwing away a badly cut dataset must cost the ENCODE, not the triage. The
    clips stay; they just stop claiming to have been promoted."""
    with app.app_context():
        bank_id, _ids, out = _promoted(app, tmp_path, seams)

        assert svc.delete_video_dataset(LOCAL_USER, out['id']) is True

        assert VideoClip.query.filter_by(bank_id=bank_id).count() == 2
        assert VideoClip.query.filter(
            VideoClip.promoted_dataset_id.isnot(None)).count() == 0


# --- wave 2: the thumbnail follows the sharpest frame ---------------------------

def _measured_bank(app, tmp_path, sharpest=None):
    """A bank with one detected clip, optionally carrying a measured sharpest
    frame. Returns (bank_id, clip_id, expected_middle)."""
    import json as _json
    from app.models import VideoClip
    with app.app_context():
        bank_id, _ = _bank(app, tmp_path, ('a.mp4',))
        svc.start_probe(app, LOCAL_USER, bank_id)
        svc.start_detect(app, LOCAL_USER, bank_id)
        clip = VideoClip.query.filter_by(bank_id=bank_id).first()
        middle = clip.start_s + (clip.end_s - clip.start_s) / 2
        if sharpest is not None:
            clip.metrics_json = _json.dumps({'metrics_state': 'ok',
                                             'sharpest_frame_s': sharpest})
            from app.extensions import db
            db.session.commit()
        return bank_id, clip.id, middle


def test_a_measured_clip_thumbnails_at_its_sharpest_frame(app, tmp_path, seams,
                                                          monkeypatch):
    """The middle-of-shot frame was a guess made before anything was measured; a
    boundary is where a cut just happened. Once the metrics scan has read every
    frame anyway, the sharpest one is a measurement — and it wins."""
    grabbed = []
    monkeypatch.setattr(svc, '_write_thumbnail',
                        lambda path, ts, out: (grabbed.append(ts), True)[1])
    bank_id, _clip_id, _middle = _measured_bank(app, tmp_path, sharpest=3.25)

    with app.app_context():
        svc.start_thumbs(app, LOCAL_USER, bank_id)

    assert 3.25 in grabbed


def test_an_unmeasured_clip_still_thumbnails_at_its_middle(app, tmp_path, seams,
                                                           monkeypatch):
    """The guess stays for clips the scan has not reached: a bank must be able to
    make thumbnails before it has ever been measured."""
    grabbed = []
    monkeypatch.setattr(svc, '_write_thumbnail',
                        lambda path, ts, out: (grabbed.append(ts), True)[1])
    bank_id, _clip_id, middle = _measured_bank(app, tmp_path, sharpest=None)

    with app.app_context():
        svc.start_thumbs(app, LOCAL_USER, bank_id)

    assert middle in grabbed


# --- the canvas cap (max_pixels) at promotion -----------------------------------

def test_source_size_promotion_is_refused_when_a_source_exceeds_the_canvas_cap(
        app, tmp_path, seams, monkeypatch):
    """MiniMax H3 caps the canvas area, and "keep the source's size" quietly
    bypasses the explicit-size validation. A 1920x1088 source is a clean multiple
    of 32 and still out of spec — encoding it would produce a dataset the model
    was never meant to ingest, with no error anywhere. Refused at launch, with
    the cap in the message, before any folder exists."""
    import pytest as _pytest
    bank_id = _ready_for_promotion(app, tmp_path, width=1920, height=1088)

    with app.app_context():
        with _pytest.raises(ValueError) as e:
            svc.start_promote(app, LOCAL_USER, bank_id, name='H3 set',
                              target_profile='minimax_h3', frames=39)
        assert 'canvas' in str(e.value) or 'max' in str(e.value)


def test_source_size_promotion_passes_when_sources_fit_the_cap(
        app, tmp_path, seams, monkeypatch):
    bank_id = _ready_for_promotion(app, tmp_path, width=768, height=1024)

    with app.app_context():
        result = svc.start_promote(app, LOCAL_USER, bank_id, name='H3 set',
                                   target_profile='minimax_h3', frames=39)
        assert result['clips'] == 1


def test_an_explicit_size_within_the_cap_is_never_blocked_by_big_sources(
        app, tmp_path, seams, monkeypatch):
    """Choosing 768x1344 RESCALES every clip, so the sources' own size stops
    mattering — the guard must only bite when the source size would survive."""
    bank_id = _ready_for_promotion(app, tmp_path, width=1920, height=1088)

    with app.app_context():
        result = svc.start_promote(app, LOCAL_USER, bank_id, name='H3 set',
                                   target_profile='minimax_h3', frames=39,
                                   size=(768, 1344))
        assert result['clips'] == 1


def _ready_for_promotion(app, tmp_path, *, width, height):
    """A bank with one probed source of the given geometry and one KEPT clip long
    enough for any profile's default length."""
    from app.extensions import db
    from app.models import VideoClip, VideoSource
    with app.app_context():
        bank_id, _ = _bank(app, tmp_path, ('a.mp4',))
        src = VideoSource.query.filter_by(bank_id=bank_id).first()
        src.probe_state = 'ok'
        src.duration_s = 60.0
        src.fps_native = 30.0
        src.width = width
        src.height = height
        clip = VideoClip(bank_id=bank_id, source_id=src.id, start_s=0.0,
                         end_s=20.0, status='keep')
        db.session.add(clip)
        db.session.commit()
        return bank_id


def test_dataset_rows_carry_a_step_suggestion_derived_from_their_own_count(app):
    """`suggested_steps` rides the same payload as `clips` and is computed from
    the same query, so the two can never disagree on screen. The launch routes
    never read it - it prefills an editable field, and what the user sends is
    what trains."""
    from app.extensions import db
    from app.models import VideoDataset, VideoDatasetClip
    from app.services import video_bank_service as svc
    from app.services import video_training
    with app.app_context():
        ds = VideoDataset(user_id='local', name='sugg', target_profile='wan22_14b',
                          fps=16, frames=81, output_dir='/tmp/x')
        db.session.add(ds)
        db.session.commit()
        for i in range(53):
            db.session.add(VideoDatasetClip(dataset_id=ds.id,
                                            filename=f'clip_{i:04d}.mp4'))
        db.session.commit()
        row = svc.video_dataset_payload('local', ds.id)
        assert row['clips'] == 53
        assert row['suggested_steps'] == video_training.suggested_steps(53) == 1500


def test_the_trigger_rides_every_sidecar_exactly_once():
    """One trigger, one place: prepended at sidecar-WRITE time, so the stored
    caption in DB and on screen stays clean, and re-editing a caption keeps the
    trigger without ever doubling it. Doubling is not cosmetic - fal measured a
    duplicated trigger degrading prompt adherence. Idempotent on captions that
    already start with it; absent, the sidecar is the caption verbatim."""
    from app.services.video_bank_service import _with_trigger
    assert _with_trigger('mychar', 'a woman walks') == 'mychar, a woman walks'
    assert _with_trigger('mychar', 'Mychar, a woman walks') == 'Mychar, a woman walks'
    assert _with_trigger('mychar', '') == 'mychar'
    assert _with_trigger('', 'a woman walks') == 'a woman walks'
    assert _with_trigger(None, None) == ''


def test_a_stills_set_reuses_the_image_lane_exporter_and_registers_its_rows(
        app, tmp_path, monkeypatch):
    """The image side already owns everything a stills set needs - kept images,
    edited captions, a trigger - so the creator REUSES its exporter rather than
    growing a second one, copies the face trigger onto the set (the idempotent
    sidecar prepend then guards re-edits), and registers rows so the dataset
    page shows what a promotion shows. An export that yields nothing rolls the
    whole creation back: an empty dataset row would be a promise of training."""
    from app.services import video_bank_service as svc
    from app.services import lora_training as lt

    def fake_export(user_id, dataset_id, masked=True, dest_dir=None, **kw):
        from pathlib import Path
        d = Path(dest_dir)
        (d / 'img_0001.png').write_bytes(b'x')
        (d / 'img_0001.txt').write_text('trig, a portrait', encoding='utf-8')
        (d / 'img_0002.png').write_bytes(b'x')
        return str(d)

    class FakeFace:
        id = 7
        name = 'kai'
        trigger_word = 'trig'

    monkeypatch.setattr(lt, 'export_dataset_to_aitoolkit', fake_export)
    monkeypatch.setattr(svc.cfg, 'video_datasets_root', lambda: tmp_path / 'vd')
    import app.services.face_dataset_service as fds
    monkeypatch.setattr(fds, 'get_dataset', lambda u, i: FakeFace())
    with app.app_context():
        out = svc.create_stills_dataset_from_face_dataset('local', 7)
        assert out['clips'] == 2
        row = svc.video_dataset_payload('local', out['id'])
        assert row['frames'] == 1 and row['target_profile'] == 'minimax_h3'
        assert row['trigger_word'] == 'trig'
        captions = {i['filename']: i['caption'] for i in row['items']}
        assert captions['img_0001.png'] == 'trig, a portrait'
        assert captions['img_0002.png'] is None


def test_references_cover_every_clip_or_are_refused(app, tmp_path, monkeypatch):
    """The trainer matches a control to a clip by FILE STEM inside each dir and
    a clip whose stem finds nothing trains without its reference, silently. The
    attach route forecloses that: each reference is written once per clip stem,
    replacing the previous set whole, and only the target that READS references
    accepts them at all."""
    from app.extensions import db
    from app.models import VideoDataset, VideoDatasetClip
    from app.services import video_bank_service as svc
    with app.app_context():
        out = tmp_path / 'ds'
        out.mkdir()
        ds = VideoDataset(user_id='local', name='ref', frames=39, fps=24,
                          target_profile='minimax_h3_ref2va',
                          output_dir=str(out))
        db.session.add(ds)
        db.session.commit()
        with pytest.raises(ValueError):        # no clips yet
            svc.set_dataset_references('local', ds.id, [('a.png', b'x')])
        for i in (1, 2):
            db.session.add(VideoDatasetClip(dataset_id=ds.id,
                                            filename=f'clip_{i:04d}.mp4'))
        db.session.commit()
        res = svc.set_dataset_references(
            'local', ds.id, [('face.png', b'a'), ('side.jpg', b'b')])
        assert res == {'references': 2, 'clips_covered': 2}
        ref1 = out / '_controls' / 'ref_1'
        assert sorted(f.name for f in ref1.iterdir()) == [
            'clip_0001.png', 'clip_0002.png']
        assert (out / '_controls' / 'ref_2' / 'clip_0001.jpg').read_bytes() == b'b'
        # replacing whole: a new single-ref set leaves no ref_2 behind
        svc.set_dataset_references('local', ds.id, [('new.png', b'c')])
        assert len(svc.reference_dirs(ds)) == 1
        # the payload says how many references ride the dataset
        assert svc.video_dataset_payload('local', ds.id)['references'] == 1
        # a target that does not read references refuses them
        ds.target_profile = 'minimax_h3'
        db.session.commit()
        with pytest.raises(ValueError):
            svc.set_dataset_references('local', ds.id, [('x.png', b'x')])



# --- the sidecar carries the MEASURED camera line (C12, 2026-08-30) ------------------

def _pan_scores():
    """A metrics blob the camera pass could have written: measured, panning
    right, camera clearly dominant — the STORED RATES, not label names, because
    labels() derives at read time from exactly these keys."""
    from app.services import video_camera_motion as vcm
    return {vcm.STATE_KEY: 'ok', 'camera_coverage': 1.0,
            'camera_pan_rate': vcm.PAN_FLOOR * 10}


def test_the_sidecar_gains_the_classifier_camera_line():
    import json as _json
    from app.services import video_camera_motion as vcm
    from app.services.video_bank_service import compose_sidecar_text
    scores = _pan_scores()
    phrase = vcm.camera_phrase(scores)
    assert phrase, 'fixture scores must produce a phrase for this test to mean anything'

    text = compose_sidecar_text('mychar', 'a woman walks', _json.dumps(scores))

    assert text.startswith('mychar, a woman walks')
    assert f'Camera: {phrase}.' in text


def test_no_camera_line_when_the_classifier_had_nothing_honest_to_say():
    """Unmeasured metrics, unreadable JSON, no JSON at all: the caption stays
    exactly what it was — a prompt silent about the camera teaches nothing
    false, a guessed one teaches the wrong word."""
    from app.services.video_bank_service import compose_sidecar_text

    assert compose_sidecar_text('mychar', 'a woman walks', None) \
        == 'mychar, a woman walks'
    assert compose_sidecar_text('mychar', 'a woman walks', 'not json{') \
        == 'mychar, a woman walks'
    assert compose_sidecar_text('mychar', 'a woman walks', '{"cam_state": "unreadable"}') \
        == 'mychar, a woman walks'


def test_a_camera_line_alone_is_still_a_sidecar():
    """A clip with no caption but a measured camera writes the one honest thing
    it knows rather than an empty prompt."""
    import json as _json
    from app.services.video_bank_service import compose_sidecar_text
    text = compose_sidecar_text('', None, _json.dumps(_pan_scores()))
    assert text.startswith('Camera: ')


# --- the measured Audio: line (C12-B) ------------------------------------------------

def test_audio_line_only_for_targets_that_keep_audio_and_only_when_measured():
    import json as _json
    from app.services.video_bank_service import compose_sidecar_text

    no_track = _json.dumps({'audio_state': 'none'})
    near_silent = _json.dumps({'audio_state': 'ok', 'silence_ratio': 1.0,
                               'rms_dbfs': -120.0})
    audible = _json.dumps({'audio_state': 'ok', 'silence_ratio': 0.1,
                           'rms_dbfs': -18.0})
    unreadable = _json.dumps({'audio_state': 'unreadable'})

    # For a target that keeps audio: the two sentences the numbers prove.
    assert compose_sidecar_text('', 'a scene', no_track, keeps_audio=True) \
        == 'a scene Audio: silent.'
    assert compose_sidecar_text('', 'a scene', near_silent, keeps_audio=True) \
        == 'a scene Audio: near silence.'
    # Audible audio: NO line — we have level metrics, not a content classifier,
    # and "ambient sound" would claim a content type nobody measured.
    assert compose_sidecar_text('', 'a scene', audible, keeps_audio=True) \
        == 'a scene'
    assert compose_sidecar_text('', 'a scene', unreadable, keeps_audio=True) \
        == 'a scene'
    # For a target whose export strips audio (Wan, -an): never a word about a
    # soundtrack the clip will not have.
    assert compose_sidecar_text('', 'a scene', no_track, keeps_audio=False) \
        == 'a scene'


def test_published_word_budgets_and_only_published_ones():
    """WAN's own rewriter caps at 200 (T2V) / 100 (I2V) words — published, so
    carried. H3 published none, so it carries none: an invented budget is a
    figure somebody plans around."""
    from app.services import video_targets as vt
    assert vt.get('wan22_14b').get('caption_word_budget') == 200
    assert vt.get('wan22_14b_i2v').get('caption_word_budget') == 100
    assert 'caption_word_budget' not in vt.get('minimax_h3')
