"""Re-cutting a bank without re-running the detector.

The detection pass now keeps its per-frame probabilities, and this is what that
buys: a threshold per bank and per file, a re-cut of one source or of a whole
bank straight from the cache, a dry run that says how many shots each threshold
would give, and a "this file is one single take" action for the single-take half
of a mixed corpus.

Three rules are defended harder than the rest, because breaking any of them
loses work the user cannot get back:

  * a bank-wide re-cut never touches a HAND-MADE clip, and never touches a
    PROMOTED one — same contract the detection pass already keeps;
  * every clip a re-cut replaces takes its thumbnail, its metrics and its
    search vectors with it, file included: a measurement of a span nobody has
    any more is not stale, it is false;
  * a re-cut from cache starts no subprocess. If it ever does, the feature is a
    lie and the user waits minutes for something advertised as instant.

No torch, no PyAV, no ffmpeg: the detector is a seam, and the probabilities are
synthetic vectors.
"""
import pytest

from app.config import LOCAL_USER
from app.models import VideoClip, VideoDataset, VideoSource
from app.services import shot_probs
from app.services import video_bank_service as svc


def _promote(clip):
    """Attach a clip to a REAL dataset row. `promoted_dataset_id` is a foreign
    key, so a made-up id would fail the constraint rather than exercise the
    rule the test is about."""
    dataset = VideoDataset(user_id=LOCAL_USER, name='built', target_profile='wan',
                           output_dir='/tmp/does-not-need-to-exist')
    svc.db.session.add(dataset)
    svc.db.session.flush()
    clip.promoted_dataset_id = dataset.id
    svc.db.session.commit()


def _probe(path):
    """The probe seam. It reports the file's REAL size on purpose: the cached
    re-detection uses that size as its staleness tripwire, and a fake probe that
    invented one would make every test here take the decode path while looking
    like it took the cache."""
    import os
    return {'duration_s': 4.0, 'fps_native': 25.0, 'width': 640, 'height': 360,
            'codec': 'h264', 'probe_state': 'ok',
            'file_size': os.path.getsize(path)}


def _vector(peaks, n=100):
    """A 100-frame (4 s at 25 fps) file with a transition at each peak."""
    probs = [0.0] * n
    for index, height in peaks.items():
        probs[index] = height
    return probs


def _detect(single, every=None, fps=25.0):
    """The detect seam, answering with a vector instead of a model."""
    def run(path, fps_native=None, **kwargs):
        from app.services import shot_detect as sd
        probs = {'single': single, 'all': every}
        return {'clips': sd.clips_from_probs(probs, fps_native=fps_native or fps,
                                             threshold=kwargs.get('threshold'),
                                             min_shot_frames=1),
                'probs': probs, 'fps_native': fps_native or fps,
                'frame_count': len(single)}
    return run


@pytest.fixture()
def bank(app, tmp_path, monkeypatch):
    """A probed, detected one-file bank whose source has two shots."""
    folder = tmp_path / 'rushes'
    folder.mkdir()
    (folder / 'a.mp4').write_bytes(b'\x00' * 32)
    monkeypatch.setattr(svc, '_probe_file', _probe)
    monkeypatch.setattr(svc, '_write_thumbnail', lambda *a, **k: True)
    monkeypatch.setattr(svc, '_detect_source',
                        _detect(_vector({40: 0.6, 70: 0.95})))
    with app.app_context():
        row, _added = svc.create_bank(LOCAL_USER, 'rushes', str(folder))
        bank_id = row.id
        svc.start_probe(app, LOCAL_USER, bank_id)
        svc.start_detect(app, LOCAL_USER, bank_id)
        source_id = VideoSource.query.filter_by(bank_id=bank_id).one().id
        yield bank_id, source_id


def _clips(bank_id):
    return (VideoClip.query.filter_by(bank_id=bank_id)
            .order_by(VideoClip.start_s.asc()).all())


# --- the pass fills the cache --------------------------------------------------

def test_detection_persists_the_probabilities_it_measured(app, bank):
    with app.app_context():
        bank_id, source_id = bank
        cached = shot_probs.load_probs(bank_id, source_id)
        assert cached is not None
        assert len(cached['single']) == 100
        assert VideoSource.query.get(source_id).probs_state == 'ok'


def test_a_source_with_no_cache_says_so_rather_than_pretending(app, bank,
                                                               monkeypatch):
    with app.app_context():
        bank_id, source_id = bank
        shot_probs.forget(bank_id, source_id)
        assert shot_probs.load_probs(bank_id, source_id) is None


# --- which threshold applies ---------------------------------------------------

def test_a_bank_threshold_overrides_the_global_default(app, bank):
    with app.app_context():
        bank_id, source_id = bank
        svc.set_bank_shot_threshold(LOCAL_USER, bank_id, 0.8)
        assert svc.shot_threshold_for(bank_id, source_id) == 0.8


def test_a_file_threshold_overrides_its_own_bank(app, bank):
    with app.app_context():
        bank_id, source_id = bank
        svc.set_bank_shot_threshold(LOCAL_USER, bank_id, 0.8)
        svc.set_source_shot_threshold(LOCAL_USER, bank_id, source_id, 0.4)
        assert svc.shot_threshold_for(bank_id, source_id) == 0.4


def test_clearing_an_override_falls_back_rather_than_meaning_zero(app, bank):
    """None is 'inherit', and 0.0 is a threshold that cuts everywhere. A UI that
    could not tell them apart would turn a cleared field into 300 clips."""
    with app.app_context():
        bank_id, source_id = bank
        svc.set_bank_shot_threshold(LOCAL_USER, bank_id, 0.8)
        svc.set_source_shot_threshold(LOCAL_USER, bank_id, source_id, 0.4)
        svc.set_source_shot_threshold(LOCAL_USER, bank_id, source_id, None)
        assert svc.shot_threshold_for(bank_id, source_id) == 0.8


def test_an_out_of_range_threshold_is_refused_at_the_door(app, bank):
    """Clamping silently in the setter would show the user a value they never
    typed. The read path clamps (it must never abort a pass); the WRITE path
    refuses, because there is somebody there to be told."""
    with app.app_context():
        bank_id, _source_id = bank
        with pytest.raises(ValueError):
            svc.set_bank_shot_threshold(LOCAL_USER, bank_id, 4)


# --- re-cutting from the cache --------------------------------------------------

def test_recutting_one_source_from_cache_never_starts_the_detector(app, bank,
                                                                   monkeypatch):
    with app.app_context():
        bank_id, source_id = bank

        def forbidden(*_a, **_kw):
            raise AssertionError('a re-cut from cache must not run the detector')
        monkeypatch.setattr(svc, '_detect_source', forbidden)

        out = svc.recut_source(LOCAL_USER, bank_id, source_id, threshold=0.8)

        assert out['clips'] == 2                # 0.6 peak no longer clears the bar
        assert len(_clips(bank_id)) == 2


def test_recutting_a_source_with_no_cache_says_what_to_do(app, bank):
    with app.app_context():
        bank_id, source_id = bank
        shot_probs.forget(bank_id, source_id)
        with pytest.raises(svc.ShotProbsMissing):
            svc.recut_source(LOCAL_USER, bank_id, source_id, threshold=0.8)


def test_a_recut_uses_the_files_own_threshold_when_none_is_given(app, bank):
    with app.app_context():
        bank_id, source_id = bank
        svc.set_source_shot_threshold(LOCAL_USER, bank_id, source_id, 0.8)

        svc.recut_source(LOCAL_USER, bank_id, source_id)

        assert len(_clips(bank_id)) == 2


def test_a_recut_spares_a_promoted_clip(app, bank):
    """A dataset already built keeps its provenance. Same rule the detection
    pass keeps, for the same reason."""
    with app.app_context():
        bank_id, source_id = bank
        kept = _clips(bank_id)[0]
        _promote(kept)

        svc.recut_source(LOCAL_USER, bank_id, source_id, threshold=0.8)

        assert VideoClip.query.get(kept.id) is not None


def test_a_bank_wide_recut_spares_hand_made_cuts(app, bank):
    """An afternoon of retouching must not disappear behind a slider."""
    with app.app_context():
        bank_id, source_id = bank
        hand = _clips(bank_id)[0]
        hand.detector = 'manual'
        svc.db.session.commit()

        svc.recut_bank(LOCAL_USER, bank_id, threshold=0.8)

        survivor = VideoClip.query.get(hand.id)
        assert survivor is not None and survivor.detector == 'manual'


def test_a_recut_takes_the_thumbnails_of_the_clips_it_replaced(app, bank):
    """The grid points an <img> at the thumb URL, which serves whatever is on
    disk. A leftover file would keep showing a frame the shot no longer
    contains — the exact lie this contract exists to remove."""
    with app.app_context():
        bank_id, source_id = bank
        doomed = _clips(bank_id)[0]
        path = svc.thumb_path(bank_id, doomed.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'\xff\xd8\xff')

        svc.recut_source(LOCAL_USER, bank_id, source_id, threshold=0.8)

        assert not path.exists()


def test_a_recut_leaves_the_new_clips_unmeasured_rather_than_inheriting(app, bank):
    with app.app_context():
        bank_id, source_id = bank

        svc.recut_source(LOCAL_USER, bank_id, source_id, threshold=0.8)

        for clip in _clips(bank_id):
            assert clip.thumb_state is None
            assert clip.metrics_json is None
            assert clip.embed_state is None
            assert clip.status == 'pending'


def test_recutting_a_whole_bank_answers_with_what_changed(app, bank):
    with app.app_context():
        bank_id, _source_id = bank

        out = svc.recut_bank(LOCAL_USER, bank_id, threshold=0.8)

        assert out['sources'] == 1
        assert out['clips'] == 2
        assert out['skipped'] == 0


def test_recutting_a_bank_skips_the_sources_that_have_no_cache(app, bank):
    """Sources detected before this existed have no vector. They are counted
    and named, never silently left with their old cuts as if nothing happened."""
    with app.app_context():
        bank_id, source_id = bank
        shot_probs.forget(bank_id, source_id)

        out = svc.recut_bank(LOCAL_USER, bank_id, threshold=0.8)

        assert out['skipped'] == 1 and out['sources'] == 0


# --- the dry run ----------------------------------------------------------------

def test_the_dry_run_answers_for_several_thresholds_without_cutting(app, bank):
    with app.app_context():
        bank_id, source_id = bank
        before = [(c.start_s, c.end_s) for c in _clips(bank_id)]

        out = svc.shot_dry_run(LOCAL_USER, bank_id, source_id=source_id,
                               thresholds=[0.5, 0.8])

        assert out['rows'] == [{'threshold': 0.5, 'shots': 3},
                               {'threshold': 0.8, 'shots': 2}]
        assert [(c.start_s, c.end_s) for c in _clips(bank_id)] == before


def test_the_dry_run_over_a_whole_bank_totals_its_files(app, bank):
    with app.app_context():
        bank_id, _source_id = bank

        out = svc.shot_dry_run(LOCAL_USER, bank_id, thresholds=[0.8])

        assert out['rows'] == [{'threshold': 0.8, 'shots': 2}]
        assert out['sources'] == 1


def test_the_dry_run_reports_how_many_files_it_could_not_answer_for(app, bank):
    with app.app_context():
        bank_id, source_id = bank
        shot_probs.forget(bank_id, source_id)

        out = svc.shot_dry_run(LOCAL_USER, bank_id, thresholds=[0.8])

        assert out['skipped'] == 1


def test_a_bank_preview_does_not_count_a_file_the_recut_would_skip(app, bank):
    """A preview that does not match the action it previews is worse than no
    preview. The bank-wide re-cut walks past a declared single take, so the
    bank-wide count must too — and must say it did."""
    with app.app_context():
        bank_id, source_id = bank
        svc.mark_single_shot(LOCAL_USER, bank_id, source_id)

        out = svc.shot_dry_run(LOCAL_USER, bank_id, thresholds=[0.5])

        assert out['rows'] == [{'threshold': 0.5, 'shots': 0}]
        assert out['single_shot'] == 1 and out['sources'] == 0


def test_asked_about_that_file_by_name_the_preview_answers_anyway(app, bank):
    """The per-file re-cut DOES apply to a declared single take — it is the way
    back from the declaration — so a preview of that one file must answer."""
    with app.app_context():
        bank_id, source_id = bank
        svc.mark_single_shot(LOCAL_USER, bank_id, source_id)

        out = svc.shot_dry_run(LOCAL_USER, bank_id, source_id=source_id,
                               thresholds=[0.5])

        assert out['rows'] == [{'threshold': 0.5, 'shots': 3}]


def test_a_bank_preview_marks_the_banks_own_value_not_one_files_override(app,
                                                                         bank):
    """Every other row's "8 fewer than now" is measured against the row marked
    in force. Marking a per-file override there would measure the whole bank
    against a value nothing bank-wide uses."""
    with app.app_context():
        bank_id, source_id = bank
        svc.set_bank_shot_threshold(LOCAL_USER, bank_id, 0.6)
        svc.set_source_shot_threshold(LOCAL_USER, bank_id, source_id, 0.3)

        assert svc.shot_dry_run(LOCAL_USER, bank_id)['current'] == 0.6
        assert svc.shot_dry_run(LOCAL_USER, bank_id,
                                source_id=source_id)['current'] == 0.3


def test_the_dry_run_offers_a_ladder_when_asked_for_no_thresholds(app, bank):
    with app.app_context():
        bank_id, source_id = bank

        out = svc.shot_dry_run(LOCAL_USER, bank_id, source_id=source_id)

        assert [r['threshold'] for r in out['rows']] == [0.3, 0.4, 0.5, 0.6,
                                                         0.7, 0.8]


# --- "this file is one single take" ---------------------------------------------

def test_single_shot_replaces_every_clip_of_the_file_with_one(app, bank):
    with app.app_context():
        bank_id, source_id = bank

        out = svc.mark_single_shot(LOCAL_USER, bank_id, source_id)

        clips = _clips(bank_id)
        assert len(clips) == 1
        assert clips[0].start_s == 0.0
        assert clips[0].end_s == pytest.approx(4.0)
        assert out['clips'] == 1


def test_the_single_shot_clip_is_hand_made_so_a_bank_recut_leaves_it_alone(app,
                                                                          bank):
    """Declaring a file a single take is a decision, and a bank-wide slider is
    not allowed to overrule a decision."""
    with app.app_context():
        bank_id, source_id = bank
        svc.mark_single_shot(LOCAL_USER, bank_id, source_id)

        svc.recut_bank(LOCAL_USER, bank_id, threshold=0.4)

        clips = _clips(bank_id)
        assert len(clips) == 1 and clips[0].detector == 'manual'


def test_a_recut_counts_single_takes_apart_from_files_it_could_not_answer_for(
        app, bank):
    """"You told me not to" and "this file has no cache" are different
    outcomes. Merging them would offer the user a fix for something that is not
    broken."""
    with app.app_context():
        bank_id, source_id = bank
        svc.mark_single_shot(LOCAL_USER, bank_id, source_id)

        out = svc.recut_bank(LOCAL_USER, bank_id, threshold=0.4)

        assert out['single_shot'] == 1 and out['skipped'] == 0


def test_the_detection_pass_itself_skips_a_declared_single_take(app, bank):
    """Not just the re-cut: 'Find shots again' is a bulk gesture too, and a
    declaration is not something a bulk gesture may overrule."""
    with app.app_context():
        bank_id, source_id = bank
        svc.mark_single_shot(LOCAL_USER, bank_id, source_id)

        svc.start_detect(app, LOCAL_USER, bank_id, redetect=True)

        assert len(_clips(bank_id)) == 1


def test_find_shots_again_reuses_the_measurement_it_already_paid_for(app, bank,
                                                                     monkeypatch):
    """The expensive half of that pass is decoding, and it has already been paid
    for. Re-running it over a bank that has been through it once must not decode
    a single frame."""
    with app.app_context():
        bank_id, _source_id = bank

        def forbidden(*_a, **_kw):
            raise AssertionError('Find shots again must not re-decode a cached file')
        monkeypatch.setattr(svc, '_detect_source', forbidden)

        svc.start_detect(app, LOCAL_USER, bank_id, redetect=True)

        assert len(_clips(bank_id)) == 3
        # And it SAYS it reused them. A pass that silently finished in a second
        # where it used to take minutes reads as a pass that did nothing.
        assert 're-cut from cache' in (svc.activity(bank_id) or {}).get('detail', '')


def test_a_source_file_that_changed_on_disk_is_decoded_again(app, bank,
                                                             monkeypatch):
    """A bank points at a LIVE folder, and people overwrite files in it. A cache
    keyed on nothing but the source id would re-cut the NEW file at the OLD
    file's boundaries and report a clean run."""
    with app.app_context():
        bank_id, source_id = bank
        source = VideoSource.query.get(source_id)
        source.file_size = (source.file_size or 32) + 1024   # as if re-exported
        svc.db.session.commit()
        seen = []

        def spy(path, fps_native=None, **kwargs):
            seen.append(path)
            return _detect(_vector({40: 0.6, 70: 0.95}))(path, fps_native, **kwargs)
        monkeypatch.setattr(svc, '_detect_source', spy)

        svc.start_detect(app, LOCAL_USER, bank_id, redetect=True)

        assert len(seen) == 1


def test_a_file_with_no_cache_is_decoded_rather_than_skipped(app, bank,
                                                             monkeypatch):
    with app.app_context():
        bank_id, source_id = bank
        shot_probs.forget(bank_id, source_id)
        seen = []

        def spy(path, fps_native=None, **kwargs):
            seen.append(path)
            return _detect(_vector({40: 0.6, 70: 0.95}))(path, fps_native, **kwargs)
        monkeypatch.setattr(svc, '_detect_source', spy)

        svc.start_detect(app, LOCAL_USER, bank_id, redetect=True)

        assert len(seen) == 1


def test_a_cached_re_detection_still_spares_hand_made_cuts(app, bank):
    with app.app_context():
        bank_id, _source_id = bank
        hand = _clips(bank_id)[0]
        hand.detector = 'manual'
        svc.db.session.commit()

        svc.start_detect(app, LOCAL_USER, bank_id, redetect=True)

        assert VideoClip.query.get(hand.id) is not None


def test_an_explicit_re_detection_of_that_one_file_undoes_it(app, bank):
    """The way back. A per-file re-cut is a deliberate gesture on a file the
    user picked, so it replaces hand-made cuts too — which a bank-wide pass
    must never do. That asymmetry is the whole design."""
    with app.app_context():
        bank_id, source_id = bank
        svc.mark_single_shot(LOCAL_USER, bank_id, source_id)

        out = svc.recut_source(LOCAL_USER, bank_id, source_id, threshold=0.5)

        assert out['clips'] == 3
        assert out['replaced_manual'] == 1


def test_single_shot_needs_a_probed_duration_and_says_so(app, bank):
    with app.app_context():
        bank_id, source_id = bank
        source = VideoSource.query.get(source_id)
        source.duration_s = None
        svc.db.session.commit()

        with pytest.raises(ValueError):
            svc.mark_single_shot(LOCAL_USER, bank_id, source_id)


def test_single_shot_spares_a_promoted_clip_like_everything_else(app, bank):
    with app.app_context():
        bank_id, source_id = bank
        kept = _clips(bank_id)[0]
        _promote(kept)

        svc.mark_single_shot(LOCAL_USER, bank_id, source_id)

        assert VideoClip.query.get(kept.id) is not None


# --- what the payload carries ---------------------------------------------------

def test_a_source_row_says_whether_it_can_be_re_cut_instantly(app, bank):
    with app.app_context():
        bank_id, source_id = bank
        rows = svc.sources_payload(LOCAL_USER, bank_id)
        assert rows[0]['has_probs'] is True
        assert rows[0]['shot_threshold'] is None

        svc.set_source_shot_threshold(LOCAL_USER, bank_id, source_id, 0.7)

        assert svc.sources_payload(LOCAL_USER, bank_id)[0]['shot_threshold'] == 0.7


def test_the_bank_payload_carries_the_threshold_in_force(app, bank):
    with app.app_context():
        bank_id, _source_id = bank
        svc.set_bank_shot_threshold(LOCAL_USER, bank_id, 0.65)

        payload = svc.bank_payload(LOCAL_USER, bank_id)

        assert payload['shot_detect']['threshold'] == 0.65
        assert payload['shot_detect']['default'] == 0.5


def test_a_clip_row_carries_the_transition_at_each_of_its_ends(app, tmp_path,
                                                               monkeypatch):
    single = _vector({40: 0.9})
    every = [0.0] * 100
    for i in range(32, 50):
        every[i] = 0.7
    folder = tmp_path / 'rushes2'
    folder.mkdir()
    (folder / 'a.mp4').write_bytes(b'\x00' * 32)
    monkeypatch.setattr(svc, '_probe_file', _probe)
    monkeypatch.setattr(svc, '_detect_source', _detect(single, every))
    with app.app_context():
        row, _added = svc.create_bank(LOCAL_USER, 'rushes2', str(folder))
        svc.start_probe(app, LOCAL_USER, row.id)
        svc.start_detect(app, LOCAL_USER, row.id)

        clips = svc.list_clips(LOCAL_USER, row.id)['clips']

        assert clips[0]['transition']['end'] == {'kind': 'dissolve', 'width': 18}
        assert clips[0]['transition']['start'] is None
