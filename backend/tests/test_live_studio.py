"""🔴 The live channel, piece by piece — none of it needs a GPU, ffmpeg or ComfyUI.

The pure helpers (scenes, the playback-rate arithmetic, the ffmpeg command, the
HLS playlist) are exercised directly; the session is driven step by step —
submit, complete, encode — with the queue and ffmpeg stubbed, so the ORDER of
things (prefill before the rate, the file kept while it waits, the viewer's
position bounding the producer) is what the tests pin, not thread timing.
"""
import os
import queue

import os
import pathlib

import pytest

from app.services import live_studio as live


# --- scenes -------------------------------------------------------------------

def test_scenes_are_blocks_between_separator_lines():
    text = "  first scene\nline two\n---\n\nsecond\n  ---  \n\n---\nthird\n"
    assert live.parse_scenes(text) == ['first scene\nline two', 'second', 'third']
    assert live.parse_scenes('') == []
    assert live.parse_scenes(None) == []
    assert len(live.parse_scenes(live.DEFAULT_SCENES)) == 3


def test_every_name_slot_becomes_the_subject():
    assert live.fill_scene('{NAME} meets {NAME2} and {NAME}', 'Jessy') == 'Jessy meets Jessy and Jessy'
    assert live.fill_scene('{NAME} walks', '   ') == 'a person walks'
    assert live.fill_scene('no slot here', 'x') == 'no slot here'


# --- the playback-rate arithmetic ----------------------------------------------

def test_the_sustainable_rate_is_frames_per_second_of_render():
    assert live.sustain_fps(362, 19.2) == pytest.approx(18.85, abs=0.01)
    assert live.sustain_fps(124, 0) is None
    assert live.sustain_fps('x', 3) is None
    assert live.sustain_fps(0, 3) is None


def test_auto_picks_a_tenth_under_what_the_card_sustains_within_bounds():
    # The write-up's own numbers: 362 frames in 19.2 s sustains 18.9 → auto 16.
    assert live.auto_fps(362, 19.2) == 16.0
    # A slow card is clamped at the floor, a fast one at 24.
    assert live.auto_fps(124, 60) == live.FPS_MIN
    assert live.auto_fps(362, 5) == live.FPS_MAX
    assert live.auto_fps(124, None) == live.FPS_MIN


def test_the_verdict_says_keeping_up_behind_or_measuring():
    assert live.verdict(124, 18, None, 0)['pace'] == 'measuring'
    v = live.verdict(362, 18, 19.2, 0)
    assert v['pace'] == 'keeping_up' and v['margin_seconds'] > 0 and v['sustain_fps'] == 18.9
    v = live.verdict(362, 24, 19.2, 3)
    assert v['pace'] == 'behind' and v['margin_seconds'] < 0
    assert v['runway_clips'] == int(3 * (362 / 24) / (19.2 - 362 / 24))
    assert live.verdict(362, 24, 19.2, 0)['runway_clips'] is None


# --- ffmpeg: the stretch and the segment command --------------------------------

def test_audio_is_stretched_by_rubberband_when_present_else_staged_atempo():
    assert live.audio_stretch(0.75, True) == 'rubberband=tempo=0.75000000:transients=smooth'
    assert live.audio_stretch(0.75, False) == 'atempo=0.75000000'
    # Under atempo's floor of 0.5 the slow-down is split so each stage stays above it.
    assert live.audio_stretch(0.25, False) == 'atempo=0.50000000,atempo=0.50000000'
    assert live.audio_stretch(1.0, False) == 'atempo=1.00000000'


def test_the_segment_command_retimes_offsets_and_keeps_one_pid_layout():
    cmd = live.retime_command('ffmpeg', 'in.mp4', 'out.ts', 18, 30.5, True)
    joined = ' '.join(cmd)
    assert cmd[0] == 'ffmpeg' and cmd[-1] == 'out.ts' and cmd[-2] == 'mpegts'
    assert 'setpts=PTS/0.750000' in joined          # 18 / 24
    assert 'rubberband=tempo=0.75000000' in joined
    assert '-r 18' in joined and '-g 18' in joined and '-keyint_min 18' in joined
    assert '-output_ts_offset 30.500' in joined
    assert '-streamid 0:256 -streamid 1:257' in joined
    silent = ' '.join(live.retime_command('ffmpeg', 'in.mp4', 'out.ts', 12, 0, False, with_audio=False))
    assert '-an' in silent and '[0:a]' not in silent


def test_the_playlist_follows_a_sliding_window_and_ends_only_when_told():
    text = live.playlist_text([(7, 'seg_000007.ts', 6.889), (8, 'seg_000008.ts', 6.889)])
    lines = text.splitlines()
    assert lines[0] == '#EXTM3U'
    assert '#EXT-X-MEDIA-SEQUENCE:7' in lines
    assert '#EXT-X-TARGETDURATION:7' in lines
    assert lines[-2:] == ['#EXTINF:6.889,', 'seg/seg_000008.ts'], 'relative to the playlist URL'
    assert '#EXT-X-ENDLIST' not in text
    assert live.playlist_text([], ended=True).splitlines()[-1] == '#EXT-X-ENDLIST'
    assert '#EXT-X-MEDIA-SEQUENCE:0' in live.playlist_text([])


def test_a_query_given_to_the_playlist_is_handed_to_every_segment_uri():
    text = live.playlist_text([(1, 'seg_000001.ts', 9.333), (2, 'seg_000002.ts', 9.333)])
    out = live.playlist_with_query(text, {'token': 'a b&c'})
    uris = [l for l in out.splitlines() if l and not l.startswith('#')]
    assert uris == ['seg/seg_000001.ts?token=a+b%26c', 'seg/seg_000002.ts?token=a+b%26c']
    tags = lambda t: [l for l in t.splitlines() if l.startswith('#')]  # noqa: E731
    assert tags(out) == tags(text)
    assert live.playlist_with_query(text, {}) == text


@pytest.mark.parametrize('name, ok', [
    ('seg_000001.ts', True), ('seg_123456.ts', True),
    ('seg_1.ts', False), ('../seg_000001.ts', False), ('seg_000001.ts/../x', False),
    ('stream.m3u8', False), ('', False), (None, False),
])
def test_only_the_segment_shape_this_module_writes_is_a_segment_name(name, ok):
    assert live.is_segment_name(name) is ok


# --- the session, step by step -------------------------------------------------

@pytest.fixture
def stubbed(app, monkeypatch, tmp_path):
    """A session whose queue, ComfyUI probes and ffmpeg are all stand-ins."""
    from app.job_queue import queue_manager
    from app.services import video_test_studio as vts
    monkeypatch.setattr(live, 'ffmpeg_facts', lambda force=False: {'path': 'ffmpeg', 'rubberband': True})
    monkeypatch.setattr(vts, 'registered_classes', lambda: {'PathchSageAttentionKJ'})
    monkeypatch.setattr(vts, 'eros_on_disk', lambda: False)
    jobs = []

    def fake_add_job(**kw):
        jobs.append(kw)
        return f'job-{len(jobs)}'
    monkeypatch.setattr(queue_manager, 'add_job', fake_add_job)
    cancelled = []
    monkeypatch.setattr(queue_manager, 'cancel_job', lambda job_id, user_id=None, **k: cancelled.append(job_id) or True)
    encodes = []

    def fake_run(cmd, **kw):
        encodes.append(cmd)
        with open(cmd[-1], 'wb') as fh:
            fh.write(b'TS')

        class R:
            returncode = 0
            stderr = ''
        return R()
    monkeypatch.setattr(live.subprocess, 'run', fake_run)
    sweep_output = live._sweep_comfy_output
    monkeypatch.setattr(live, '_sweep_comfy_output', lambda: None)   # never this machine's ComfyUI folder
    live._session = None
    live._recent.clear()
    return {'jobs': jobs, 'cancelled': cancelled, 'encodes': encodes, 'sweep_output': sweep_output}


def _params(**over):
    p = {'frames': 124, 'megapixels': 0.3, 'aspect': 'landscape', 'fps': 0, 'steps': 4,
         'turbo': True, 'scenes': live.DEFAULT_SCENES, 'subject': 'Jessy', 'seed': 10}
    p.update(over)
    return p


def test_a_channel_needs_scenes_and_ffmpeg_and_runs_alone(app, stubbed, monkeypatch):
    with app.app_context():
        with pytest.raises(live.LiveError):
            live.LiveSession(app, 'local', _params(scenes='   '))
        s = live.start(app, 'local', _params())
        assert s.state in ('starting', 'running')
        with pytest.raises(live.LiveError):
            live.start(app, 'local', _params())
        live.stop()
        monkeypatch.setattr(live, 'ffmpeg_facts', lambda force=False: {'path': None, 'rubberband': False})
        with pytest.raises(live.LiveError):
            live.start(app, 'local', _params())


def test_the_producer_keeps_the_pipeline_full_and_no_more(app, stubbed):
    with app.app_context():
        s = live.LiveSession(app, 'local', _params())
        assert s._may_submit()
        s._submit_one()
        s._submit_one()
        assert len(stubbed['jobs']) == live.PIPELINE
        assert not s._may_submit(), 'two prompts in the queue is the ceiling'
        md = stubbed['jobs'][0]['metadata']
        assert md['is_live'] is True and md['model_name'] == live.JOB_NAME
        assert md['live_session'] == s.id and md['live_seq'] == 1
        # The graph is the Studio's own t2v graph, with the subject in the prompt.
        wf = stubbed['jobs'][0]['workflow_data']
        assert 'Jessy' in wf['104']['inputs']['prompt'] and 'first_frame' not in wf['104']['inputs']
        assert wf['9']['inputs']['steps'] == 4
        # Seeds walk up from the session's seed, one per SUBMISSION.
        assert [j['metadata']['seed'] for j in stubbed['jobs']] == [10, 11]


def test_the_viewer_position_bounds_how_far_ahead_the_producer_renders(app, stubbed):
    with app.app_context():
        s = live.LiveSession(app, 'local', _params(fps=12))
        # Pretend the pipeline is empty and BUFFER_AHEAD segments already wait.
        s.segments = [(i, live.segment_name(i), 10.0) for i in range(1, live.BUFFER_AHEAD + live.PIPELINE + 1)]
        assert not s._may_submit(), 'nobody has watched anything: the prefill is bounded'
        s.note_segment_request(live.segment_name(4))
        assert s._may_submit(), 'the viewer moved on, there is room to render again'


def test_a_finished_clip_waits_for_the_rate_then_becomes_a_segment(app, stubbed, tmp_path):
    with app.app_context():
        s = live.LiveSession(app, 'local', _params(fps=0))      # AUTO
        s._submit_one()
        s._submit_one()
        s._submit_one.__func__  # noqa: B018 — silence "unused" on the bound method above

        def claim(filename, seq):
            p = s.dir / f'clip_{seq:06d}.mp4'
            p.write_bytes(b'MP4')
            return str(p)
        s._claim = claim
        # First clip: the card was cold (30 s), the second is the pace (12 s).
        s.on_completed('job-1', 'a.mp4', False, None, 30.0)
        assert s.pending.qsize() == 1 and s.inflight == {'job-2': 2}
        facts = live.ffmpeg_facts()
        seq, path, rs = s.pending.get()
        assert s._encode(seq, path, rs, facts) is False, 'AUTO needs the prefill before it can retime'
        assert os.path.exists(path), 'a clip put back to wait keeps its file'
        assert s.play_fps is None and s.pending.qsize() == 1

        s.on_completed('job-2', 'b.mp4', False, None, 12.0)
        # The cold first clip is excluded from the pace once another exists:
        # 124 frames / 12 s = 10.3 fps → a tenth under → 9.
        seq, path, rs = s.pending.get()
        assert s._encode(seq, path, rs, facts) is True
        assert s.play_fps == 9.0
        assert s.segments == [(1, 'seg_000001.ts', 124 / 9)]
        assert (s.dir / 'seg_000001.ts').exists() and s.playlist_path.exists()
        text = s.playlist_path.read_text(encoding='utf-8')
        assert '#EXT-X-MEDIA-SEQUENCE:1' in text and 'seg_000001.ts' in text
        assert '#EXT-X-ENDLIST' not in text
        cmd = stubbed['encodes'][-1]
        assert '-output_ts_offset' in cmd and cmd[cmd.index('-output_ts_offset') + 1] == '0.000'
        # The next segment starts where this one ends.
        seq, path, rs = s.pending.get()
        assert s._encode(seq, path, rs, facts) is True
        cmd = stubbed['encodes'][-1]
        assert cmd[cmd.index('-output_ts_offset') + 1] == f'{124 / 9:.3f}'
        st = s.status()
        assert st['produced'] == 2 and st['play_fps'] == 9.0 and st['state'] == 'starting'
        assert st['pace'] == 'keeping_up', 'the session state and the pace are two different keys'
        assert st['sustain_fps'] == round(124 / 12, 1) and st['render_seconds'] == 12.0


def test_a_user_rate_is_kept_and_a_failed_clip_is_counted_not_fed(app, stubbed):
    with app.app_context():
        s = live.LiveSession(app, 'local', _params(fps=12))
        s._submit_one()
        s.on_completed('job-1', None, True, 'ComfyUI KSampler: boom', 4.0)
        assert s.failed == 1 and s.pending.empty() and s.error == 'ComfyUI KSampler: boom'
        assert s.play_fps == 12.0 and s.auto is False
        s.on_completed('job-unknown', 'x.mp4', False, None, 1.0)
        assert s.pending.empty(), 'a completion for a job this channel never queued is ignored'


def test_a_whole_clip_clears_the_last_error_and_the_failed_count_keeps_it(app, stubbed):
    """Seen on the real engine: the first submit after a start was refused
    (ComfyUI not yet probed), the channel went on fine, and the status kept
    shouting that refusal for its whole life. A clip that comes back whole is
    the proof the pipeline works again."""
    with app.app_context():
        s = live.LiveSession(app, 'local', _params(fps=12))
        s._submit_one()
        s._submit_one()
        s.error = 'COMFYUI_UNREACHABLE (nothing was submitted)'   # what the producer loop records
        s.on_completed('job-1', None, True, 'ComfyUI KSampler: boom', 4.0)
        assert s.error == 'ComfyUI KSampler: boom' and s.failed == 1

        def claim(filename, seq):
            p = s.dir / f'clip_{seq:06d}.mp4'
            p.write_bytes(b'MP4')
            return str(p)
        s._claim = claim
        s.on_completed('job-2', 'b.mp4', False, None, 9.0)
        assert s.error is None and s.failed == 1 and s.pending.qsize() == 1


def test_the_cold_clip_is_the_first_measured_one_not_clip_number_one(app, stubbed):
    """Refuter's witness: clip #1 refused at submit, #2 cold (40 s), #3 the pace
    (20 s). Keyed on the number the mean was 40 s and the rate halved for good."""
    with app.app_context():
        s = live.LiveSession(app, 'local', _params(fps=0))
        s.render_times = {2: 40.0, 3: 20.0}
        assert s._mean_render_locked() == 20.0
        s.render_times = {2: 40.0}
        assert s._mean_render_locked() == 40.0, 'alone, the only clip is the measure'


def test_a_stop_in_auto_before_the_second_clip_still_drains_and_ends(app, stubbed):
    """Refuter's witness: AUTO (the UI default), one clip waiting for the rate,
    stop → the feeder looped forever, the state never reached stopped and the
    next start was refused for the life of the process."""
    with app.app_context():
        s = live.LiveSession(app, 'local', _params(fps=0))
        s._submit_one()
        s._submit_one()

        def claim(filename, seq):
            p = s.dir / f'clip_{seq:06d}.mp4'
            p.write_bytes(b'MP4')
            return str(p)
        s._claim = claim
        s.on_completed('job-1', 'a.mp4', False, None, 30.0)
        assert s.play_fps is None and s.pending.qsize() == 1
        s.stop()
        s._feed()                                    # the feeder thread's body, run here
        assert s.state == 'stopped' and s.pending.empty()
        assert s.play_fps == live.FPS_MIN, 'one 30 s clip of 124 frames sustains 4 fps → the floor'
        assert s.segments == [(1, 'seg_000001.ts', 124 / live.FPS_MIN)]
        assert (s.dir / 'seg_000001.ts').exists() and not (s.dir / 'seg_000001.ts.tmp').exists()
        assert s.playlist_path.read_text(encoding='utf-8').rstrip().endswith('#EXT-X-ENDLIST')


def test_the_producer_closes_the_channel_after_repeated_refused_submits(app, stubbed, monkeypatch):
    """Refuter's measure: with ComfyUI down the loop wrote a stack every 1.5 s
    (2 500 an hour, the log rotated away in ~3 h) and stayed 'starting' forever."""
    from app.job_queue import queue_manager
    calls = []

    def refuse(**kw):
        calls.append(kw)
        raise RuntimeError('COMFYUI_UNREACHABLE (nothing was submitted)')
    monkeypatch.setattr(queue_manager, 'add_job', refuse)
    monkeypatch.setattr(live, 'SUBMIT_BACKOFF', (0.01, 0.01, 0.01, 0.01, 0.01))
    with app.app_context():
        s = live.LiveSession(app, 'local', _params(fps=12))
        s._produce()                                  # returns by itself
        assert len(calls) == live.SUBMIT_RETRIES
        assert s.stop_event.is_set() and s.state == 'stopping'
        assert s.error.startswith('channel stopped') and 'COMFYUI_UNREACHABLE' in s.error
        s._feed()
        assert s.state == 'stopped'


def test_the_status_says_when_rendering_waits_for_the_viewer(app, stubbed):
    with app.app_context():
        s = live.LiveSession(app, 'local', _params(fps=12))
        s.state = 'running'
        assert s.status()['paused_for_viewer'] is False
        for i in range(1, 5):
            s.segments.append((i, live.segment_name(i), 10.0))
        s._submit_one()
        s._submit_one()                               # 4 buffered + 2 in flight, viewer at 0
        assert s._may_submit() is False and s.status()['paused_for_viewer'] is True
        s.note_segment_request('seg_000002.ts')
        assert s.status()['paused_for_viewer'] is False   # the pipeline is still full, the viewer no longer holds it


def test_a_second_stop_does_not_revive_a_channel_already_gone(app, stubbed):
    with app.app_context():
        s = live.LiveSession(app, 'local', _params(fps=12))
        s.stop()
        s._feed()
        assert s.state == 'stopped'
        s.stop()
        assert s.state == 'stopped', "a stopped channel stays stopped — 'stopping' forever blocked every next start"
        live._session = s
        assert live.stop()['state'] == 'stopped'


def test_a_clip_submitted_while_stopping_is_given_back(app, stubbed):
    """stop() snapshots inflight; a submit that slipped in between the check and
    the registration is cancelled by the producer itself."""
    with app.app_context():
        s = live.LiveSession(app, 'local', _params(fps=12))
        s.stop_event.set()
        s._submit_one()
        assert s.inflight == {} and stubbed['cancelled'] == ['job-1']


def test_segments_are_numbered_contiguously_even_when_a_clip_fails(app, stubbed):
    """HLS numbers segments: clip #2 failing must not leave a hole between
    seg 1 and seg 3, or a player reloading across it counts one off."""
    with app.app_context():
        s = live.LiveSession(app, 'local', _params(fps=12))
        for _ in range(3):
            s._submit_one()

        def claim(filename, seq):
            p = s.dir / f'clip_{seq:06d}.mp4'
            p.write_bytes(b'MP4')
            return str(p)
        s._claim = claim
        s.on_completed('job-1', 'a.mp4', False, None, 9.0)
        s.on_completed('job-2', None, True, 'boom', 9.0)
        s.on_completed('job-3', 'c.mp4', False, None, 9.0)
        s.stop()
        s._feed()
        assert [x[0] for x in s.segments] == [1, 2] and s.failed == 1
        assert [x[1] for x in s.segments] == ['seg_000001.ts', 'seg_000002.ts']
        text = s.playlist_path.read_text(encoding='utf-8')
        assert '#EXT-X-MEDIA-SEQUENCE:1' in text and 'seg/seg_000002.ts' in text and 'seg_000003' not in text


def test_a_clip_without_audio_is_encoded_video_only_behind_a_discontinuity(app, stubbed, monkeypatch):
    """The graph always makes an audio track; the day it does not, the second
    ffmpeg pass (-an) takes over and the playlist tells the player the streams
    changed — both ways."""
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        silent = '-an' in cmd
        has_audio_clip = 'audio' in os.path.basename(cmd[cmd.index('-i') + 1])

        class R:
            returncode = 0 if (silent or has_audio_clip) else 1
            stderr = '' if returncode == 0 else 'Stream map [0:a] matches no streams'
        if R.returncode == 0:
            with open(cmd[-1], 'wb') as fh:
                fh.write(b'TS')
        return R()
    monkeypatch.setattr(live.subprocess, 'run', run)
    with app.app_context():
        s = live.LiveSession(app, 'local', _params(fps=12))
        facts = live.ffmpeg_facts()
        for name in ('clip_000001_silent.mp4', 'clip_000002_audio.mp4', 'clip_000003_audio.mp4'):
            (s.dir / name).write_bytes(b'MP4')
        assert s._encode(1, str(s.dir / 'clip_000001_silent.mp4'), 9.0, facts) is True
        assert len(calls) == 2 and '-an' in calls[-1], 'the audio pass failed, the video-only pass took over'
        assert s._encode(2, str(s.dir / 'clip_000002_audio.mp4'), 9.0, facts) is True
        assert s._encode(3, str(s.dir / 'clip_000003_audio.mp4'), 9.0, facts) is True
        assert s.discontinuities == {1, 2}, 'silent first, audio back on the second — the third changes nothing'
        lines = s.playlist_path.read_text(encoding='utf-8').splitlines()
        assert lines.count('#EXT-X-DISCONTINUITY') == 2
        assert lines[lines.index('seg/seg_000002.ts') - 2] == '#EXT-X-DISCONTINUITY'
        assert lines[lines.index('seg/seg_000003.ts') - 2] != '#EXT-X-DISCONTINUITY'


def test_an_encode_failure_is_counted_and_the_channel_still_ends(app, stubbed, monkeypatch):
    with app.app_context():
        s = live.LiveSession(app, 'local', _params(fps=12))
        s.pending.put((1, str(s.dir / 'clip_000001.mp4'), 9.0))
        (s.dir / 'clip_000001.mp4').write_bytes(b'MP4')
        monkeypatch.setattr(s, '_encode', lambda *a, **k: (_ for _ in ()).throw(RuntimeError(r'C:\Users\someone\x.mp4: bad')))
        s.stop()
        s._feed()
        assert s.failed == 1 and s.state == 'stopped'
        assert s.error.startswith('encode failed')
        assert 'someone' not in s.status()['error'], 'the status is paste-safe: no account name from ffmpeg'


def test_the_stamped_job_name_is_a_literal_the_harvest_guard_can_read():
    """test_dataset_job_harvest discovers stamped names by AST; a constant would
    leave the video_live entry of its list inert."""
    src = (pathlib.Path(live.__file__)).read_text(encoding='utf-8')
    assert "'model_name': 'video_live'" in src and live.JOB_NAME == 'video_live'


def test_the_status_counts_measured_clips_apart_from_encoded_segments(app, stubbed):
    with app.app_context():
        s = live.LiveSession(app, 'local', _params(fps=0))
        s._submit_one()
        s.on_completed('job-1', None, True, 'boom', 5.0)
        s.render_times[1] = 30.0
        st = s.status()
        assert st['measured'] == 1 and st['produced'] == 0


def test_segments_behind_the_viewer_are_pruned_never_ahead_of_it(app, stubbed):
    with app.app_context():
        s = live.LiveSession(app, 'local', _params(fps=12))
        for i in range(1, 8):
            (s.dir / live.segment_name(i)).write_bytes(b'TS')
            s.segments.append((i, live.segment_name(i), 10.0))
        s.note_segment_request('seg_000006.ts')
        s.note_segment_request('seg_000003.ts')          # a retry of an older one never moves it back
        assert s.last_requested_seq == 6
        with s.lock:
            s._prune_locked()
        kept = [seq for seq, _n, _s in s.segments]
        assert kept == [4, 5, 6, 7], 'SEGMENT_KEEP behind the viewer, everything ahead'
        assert not (s.dir / 'seg_000003.ts').exists() and (s.dir / 'seg_000004.ts').exists()


def test_stop_gives_back_the_pending_jobs_and_ends_the_playlist(app, stubbed):
    with app.app_context():
        s = live.LiveSession(app, 'local', _params(fps=12))   # no threads: driven by hand
        live._session = s
        s._submit_one()
        s._submit_one()
        s._job_status = lambda job_id: 'pending'                # both still waiting in the queue
        out = live.stop()
        assert out['state'] == 'stopping'
        assert set(stubbed['cancelled']) == {'job-1', 'job-2'}
        assert s.inflight == {}
        s._feed()                                             # what the feeder thread does last
        assert s.state == 'stopped'
        assert s.playlist_path.read_text(encoding='utf-8').rstrip().endswith('#EXT-X-ENDLIST')


def test_stop_leaves_the_clip_on_the_card_alone_and_discards_it_when_it_lands(app, stubbed, monkeypatch, tmp_path):
    """Interrupting the prompt on the card raises the queue's recovery barrier,
    which refused the next start with a 409 — measured on the real engine. So
    the clip finishes for nobody, and its file goes."""
    with app.app_context():
        s = live.LiveSession(app, 'local', _params(fps=12))
        s._submit_one()
        s._submit_one()
        s._job_status = lambda job_id: 'sent_to_comfy' if job_id == 'job-1' else 'pending'
        s.stop()
        assert stubbed['cancelled'] == ['job-2'] and s.abandoned == {'job-1'} and s.inflight == {}
        out = tmp_path / 'comfy-out'
        out.mkdir()
        (out / 'late.mp4').write_bytes(b'MP4')
        from app.services import lora_test_studio as lts
        monkeypatch.setattr(lts, '_comfy_output_dir', lambda: str(out))
        s.on_completed('job-1', 'late.mp4', False, None, 20.0)
        assert not (out / 'late.mp4').exists() and s.pending.empty() and s.abandoned == set()
        s.on_completed('job-1', 'late.mp4', False, None, 20.0)   # twice: ignored, nothing to discard


def test_the_queue_routes_a_live_job_here_and_a_channelless_completion_is_harmless(app, monkeypatch):
    import json
    from app import job_queue
    from app.extensions import db
    from app.models import ImageGenerationQueue
    seen = {}
    monkeypatch.setattr(job_queue, '_drop_staged_inputs', lambda md: None)
    monkeypatch.setattr(live, 'link_completed_live_clip',
                        lambda *a, **k: seen.update({'args': a, 'kw': k}))
    with app.app_context():
        job = ImageGenerationQueue(job_id='job-live', user_id='1', status='completed',
                                   job_metadata=json.dumps({'is_live': True, 'model_name': live.JOB_NAME}))
        db.session.add(job)
        db.session.commit()
        job_queue._dispatch_completion(job, 'out.mp4', False)
        assert seen['args'] == ('job-live', 'out.mp4')
        assert seen['kw'] == {'failed': False, 'reason': None, 'session_id': None}
        # No channel open: the real callback logs and returns, never raises.
        monkeypatch.undo()
        live._session = None
        live.link_completed_live_clip('job-live', 'out.mp4')


def test_a_clip_landing_after_its_channel_was_replaced_reaches_that_channel(app, stubbed, monkeypatch, tmp_path):
    """Measured on the real engine: stop, restart at once, the abandoned clip
    lands — on the NEW channel, which did not know it, so the file stayed in
    ComfyUI's output folder. The completion carries the channel id."""
    with app.app_context():
        old = live.LiveSession(app, 'local', _params(fps=12))
        new = live.LiveSession(app, 'local', _params(fps=12))
        live._recent[old.id] = old
        live._recent[new.id] = new
        live._session = new
        old._submit_one()
        old._job_status = lambda job_id: 'sent_to_comfy'
        old.stop()
        assert old.abandoned == {'job-1'}
        out = tmp_path / 'comfy-out'
        out.mkdir()
        (out / 'late.mp4').write_bytes(b'MP4')
        from app.services import lora_test_studio as lts
        monkeypatch.setattr(lts, '_comfy_output_dir', lambda: str(out))
        monkeypatch.setattr(live, '_render_seconds', lambda job_id: 20.0)
        live.link_completed_live_clip('job-1', 'late.mp4', session_id=old.id)
        assert not (out / 'late.mp4').exists() and new.pending.empty() and old.abandoned == set()
        # A channel the registry no longer holds falls back to the open one.
        live.link_completed_live_clip('job-9', 'x.mp4', session_id='gone')
        assert new.pending.empty()


def test_a_new_channel_sweeps_the_live_clips_a_previous_one_left_in_comfyuis_output(app, stubbed, monkeypatch, tmp_path):
    out = tmp_path / 'comfy-out'
    out.mkdir()
    keep = ['local_lds_video_test_ab12_00001_.mp4', 'something_else.mp4',
            'local_lds_video_test_ab12_live_x_00001_.mp4', 'dataset_live_0123abcd_00001_.mp4']
    gone = ['local_lds_video_test_ab12_live_0123abcd_00001_.mp4', 'lds_video_test_ab12_live_deadbeef_00002_.webm']
    for name in keep + gone:
        (out / name).write_bytes(b'x')
    from app.services import lora_test_studio as lts
    monkeypatch.setattr(lts, '_comfy_output_dir', lambda: str(out))
    monkeypatch.setattr(live, '_sweep_comfy_output', stubbed['sweep_output'])
    with app.app_context():
        live._sweep_root()
    assert sorted(p.name for p in out.iterdir()) == sorted(keep)


def test_render_seconds_come_from_the_queue_row_like_the_studio_cards(app):
    from datetime import datetime, timedelta
    from app.extensions import db
    from app.models import ImageGenerationQueue
    with app.app_context():
        t0 = datetime(2026, 9, 3, 1, 0, 0)
        db.session.add(ImageGenerationQueue(job_id='j', user_id='local', status='completed',
                                            started_at=t0, completed_at=t0 + timedelta(seconds=47.62)))
        db.session.add(ImageGenerationQueue(job_id='c', user_id='local', status='cancelled',
                                            started_at=t0, completed_at=t0 + timedelta(hours=3)))
        db.session.commit()
        assert live._render_seconds('j') == 47.6
        assert live._render_seconds('c') is None
        assert live._render_seconds('missing') is None


def test_a_new_channel_sweeps_what_the_previous_one_left(app, stubbed):
    with app.app_context():
        root = live.live_root(create=True)
        (root / 'old').mkdir()
        (root / 'old' / 'seg_000001.ts').write_bytes(b'TS')
        s = live.start(app, 'local', _params(fps=12))
        assert not (root / 'old').exists() and s.dir.exists()
        live.stop()


def test_pending_is_fifo_by_completion(app, stubbed):
    q = queue.Queue()
    q.put((1, 'a', 1.0))
    q.put((2, 'b', 1.0))
    assert q.get()[0] == 1
