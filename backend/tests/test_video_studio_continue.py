"""⏭ Continue a clip from its last frame: the frame staged, the link stored,
the part joined behind the parent when it lands — with the sound of whichever
side has one, at one cadence, and never inside the write transaction."""
import os
import subprocess

import pytest

from app.services import video_test_studio as vts


def _png(path, w=64, h=32):
    from PIL import Image
    Image.new('RGB', (w, h), (10, 20, 30)).save(path)


# What ffmpeg prints for `-i` on a rendered clip (sound) and on a smoothed one
# (pictures only): the join reads both facts off this banner.
BANNER_AUDIO = ("Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'x.mp4':\n"
                '  Duration: 00:00:02.33, start: 0.000000, bitrate: 900 kb/s\n'
                '  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p, 736x416, 24 fps\n'
                '  Stream #0:1[0x2](und): Audio: aac (LC) (mp4a / 0x6134706D), 48000 Hz, stereo, fltp\n')
BANNER_SILENT = ('  Duration: 00:00:04.54, start: 0.000000, bitrate: 900 kb/s\n'
                 '  Stream #0:0[0x1](und): Video: h264 (High), yuv420p, 736x416, 48 fps\n')


class _R:
    def __init__(self, returncode=0, stderr=''):
        self.returncode, self.stderr = returncode, stderr


def _comfy(monkeypatch):
    monkeypatch.setattr('app.capabilities.probe_comfyui',
                        lambda: {'ok': True, 'status': 'ok', 'detail': '', 'hint': ''})


def test_the_probe_reads_sound_and_length_off_the_banner(monkeypatch):
    monkeypatch.setattr(vts, '_run_ffmpeg', lambda cmd, timeout=600: _R(1, BANNER_AUDIO))
    assert vts._probe_media('ffmpeg', 'x.mp4') == {'audio': True, 'duration': pytest.approx(2.33)}
    monkeypatch.setattr(vts, '_run_ffmpeg', lambda cmd, timeout=600: _R(1, BANNER_SILENT))
    assert vts._probe_media('ffmpeg', 'x.mp4') == {'audio': False, 'duration': pytest.approx(4.54)}
    monkeypatch.setattr(vts, '_run_ffmpeg', lambda cmd, timeout=600: _R(1, ''))
    assert vts._probe_media('ffmpeg', 'x.mp4') == {'audio': False, 'duration': None}


def test_the_join_command_drops_the_conditioning_frame_scales_and_paces_the_part():
    cmd = vts.continuation_command('ffmpeg', 'p.mp4', 'n.mp4', 'out.mp4', 736, 416, 24)
    joined = ' '.join(cmd)
    assert cmd[:2] == ['ffmpeg', '-hide_banner'] and cmd[-1] == 'out.mp4'
    assert '-i p.mp4 -i n.mp4' in joined, 'parent first, then the part'
    assert 'trim=start_frame=1' in joined, "the part's first frame is the parent's last"
    assert 'scale=736:416' in joined and 'concat=n=2:v=1:a=0[v]' in joined
    # One constant cadence on both sides: concat of 48 and 24 fps writes a
    # variable-rate file whose stated fps lies to Smooth and to the card.
    assert joined.count('fps=24') == 2
    assert 'atrim=start=0.041667' in joined and '[0:a]' in joined and 'concat=n=2:v=0:a=1[a]' in joined
    assert '-map_metadata -1' in joined, 'the generation graph tag does not describe the join'
    # A smoothed parent at 48 fps, the part rendered at 24: the trim is one PART frame.
    mixed = ' '.join(vts.continuation_command('ffmpeg', 'p.mp4', 'n.mp4', 'o.mp4', 736, 416, 48, part_fps=24))
    assert 'atrim=start=0.041667' in mixed and mixed.count('fps=48') == 2
    # Sound: a silent side is padded with silence of ITS length, so the
    # other side's sound survives — a smoothed parent has no track.
    silent_parent = ' '.join(vts.continuation_command('ffmpeg', 'p.mp4', 'n.mp4', 'o.mp4', 736, 416, 24,
                                                       parent_audio=False, parent_seconds=2.33))
    assert 'anullsrc=r=48000:cl=stereo:d=2.330[a0]' in silent_parent
    assert '[1:a]' in silent_parent and '-an' not in silent_parent
    silent_part = ' '.join(vts.continuation_command('ffmpeg', 'p.mp4', 'n.mp4', 'o.mp4', 736, 416, 24,
                                                     part_audio=False, part_seconds=0.917))
    assert '[0:a]' in silent_part and 'anullsrc=r=48000:cl=stereo:d=0.875[a1]' in silent_part
    neither = ' '.join(vts.continuation_command('ffmpeg', 'p.mp4', 'n.mp4', 'o.mp4', 736, 416, 24,
                                                parent_audio=False, part_audio=False))
    assert '-an' in neither and 'anullsrc' not in neither and '[1:a]' not in neither
    with pytest.raises(ValueError):
        vts.continuation_command('ffmpeg', 'p.mp4', 'n.mp4', 'o.mp4', 736, 416, 24, parent_audio=False)
    last = vts.last_frame_command('ffmpeg', 'p.mp4', 'last.png')
    assert '-sseof' in last and 'reverse' in ' '.join(last) and last[-1] == 'last.png'


def test_a_continuation_is_stored_and_refused_when_the_parent_is_not_done(app, monkeypatch, tmp_path):
    from app.extensions import db
    from app.job_queue import queue_manager
    from app.models import VideoTestClip
    monkeypatch.setattr(vts, 'registered_classes', lambda: {'PathchSageAttentionKJ'})
    monkeypatch.setattr(vts, 'preflight', lambda wf: None)
    monkeypatch.setattr(queue_manager, 'add_job', lambda **kw: kw.get('job_id'))
    with app.app_context():
        parent = VideoTestClip(status='done', filename='p.mp4', prompt='p', mode='t2v', frames=56, fps=24.0)
        pending = VideoTestClip(status='pending', prompt='q', mode='t2v')
        no_file = VideoTestClip(status='done', filename=None, prompt='r', mode='t2v')
        db.session.add_all([parent, pending, no_file])
        db.session.commit()
        out = vts.enqueue_clip('local', prompt='she turns and smiles', mode='t2v', frames=56,
                               megapixels=0.2, continues=parent.id)
        row = VideoTestClip.query.get(out['clip_id'])
        assert row.continues_of == parent.id
        for bad in (pending.id, no_file.id, 99999):
            with pytest.raises(ValueError):
                vts.enqueue_clip('local', prompt='she turns and smiles', mode='t2v', frames=56,
                                 megapixels=0.2, continues=bad)


def test_the_generate_route_carries_continues_and_refuses_an_unfinished_parent(client, app, monkeypatch):
    """The front sends `continues`, the service honours it — this is the one
    test that walks the route between them (a dropped key passed everything)."""
    from app.extensions import db
    from app.job_queue import queue_manager
    from app.models import VideoTestClip
    _comfy(monkeypatch)
    monkeypatch.setattr(vts, 'preflight', lambda wf: None)
    monkeypatch.setattr(queue_manager, 'add_job', lambda **kw: kw.get('job_id'))
    with app.app_context():
        parent = VideoTestClip(status='done', filename='p.mp4', prompt='p', mode='t2v', frames=56, fps=24.0)
        pending = VideoTestClip(status='pending', prompt='q', mode='t2v')
        db.session.add_all([parent, pending])
        db.session.commit()
        pid, qid = parent.id, pending.id
    r = client.post('/api/video-studio/generate', json={
        'mode': 't2v', 'prompt': 'she turns and smiles', 'frames': 56, 'continues': pid})
    assert r.status_code == 200, r.get_json()
    with app.app_context():
        assert db.session.get(VideoTestClip, r.get_json()['clip_id']).continues_of == pid
    r = client.post('/api/video-studio/generate', json={
        'mode': 't2v', 'prompt': 'she turns and smiles', 'frames': 56, 'continues': qid})
    assert r.status_code == 400 and 'not finished' in r.get_json()['error']


def test_the_last_frame_is_extracted_once_refreshed_when_stale_and_the_route_stages_it(app, client, monkeypatch, tmp_path):
    from app.extensions import db
    from app.models import VideoTestClip
    from app.services import ffmpeg_tools
    monkeypatch.setattr(vts, 'clips_dir', lambda create=True: str(tmp_path))
    monkeypatch.setattr(ffmpeg_tools, 'ffmpeg_path', lambda: 'ffmpeg')
    calls = []

    def fake_ffmpeg(cmd, timeout=600):
        calls.append(cmd)
        _png(cmd[-1])
        return _R()
    monkeypatch.setattr(vts, '_run_ffmpeg', fake_ffmpeg)
    (tmp_path / 'p.mp4').write_bytes(b'MP4')
    with app.app_context():
        parent = VideoTestClip(status='done', filename='p.mp4', prompt='p', mode='t2v', frames=56, fps=24.0)
        pending = VideoTestClip(status='pending', prompt='q', mode='t2v')
        db.session.add_all([parent, pending])
        db.session.commit()
        pid, qid = parent.id, pending.id
        png = vts.last_frame_png(pid)
        assert os.path.basename(png) == f'clip_{pid}_last.png' and os.path.isfile(png)
        assert vts.last_frame_png(pid) == png and len(calls) == 1, 'extracted once, kept'
        # An mp4 newer than its cache is read again: SQLite reuses a deleted
        # clip's id, and a sidecar under that id must not be served as-is.
        t = os.path.getmtime(str(tmp_path / 'p.mp4')) - 10
        os.utime(png, (t, t))
        assert vts.last_frame_png(pid) == png and len(calls) == 2, 'a newer mp4 than the cache is re-read'
        # ffmpeg that writes nothing, and no ffmpeg at all: a sentence, not a trace.
        os.remove(png)
        monkeypatch.setattr(vts, '_run_ffmpeg', lambda cmd, timeout=600: _R(0, ''))
        with pytest.raises(ValueError, match='could not be read'):
            vts.last_frame_png(pid)
        monkeypatch.setattr(ffmpeg_tools, 'ffmpeg_path', lambda: None)
        with pytest.raises(ValueError, match='ffmpeg is needed'):
            vts.last_frame_png(pid)
        monkeypatch.setattr(ffmpeg_tools, 'ffmpeg_path', lambda: 'ffmpeg')
        monkeypatch.setattr(vts, '_run_ffmpeg', fake_ffmpeg)
    staged = {}

    def stage(src, dest, input_dir):
        staged['src'] = src
        _png(os.path.join(str(tmp_path), dest))
        return os.path.join(str(tmp_path), dest)
    from app.utils import comfy_fs
    monkeypatch.setattr(comfy_fs, 'ensure_input_usable', lambda d: str(tmp_path))
    monkeypatch.setattr(comfy_fs, 'stage_input_image', stage)
    r = client.post(f'/api/video-studio/clip/{pid}/last-frame')
    body = r.get_json()
    assert r.status_code == 200 and body['ok'] and body['continues'] == pid
    assert body['image'].startswith('lds_vstudio_') and body['ratio'] == 2.0
    assert staged['src'] == png and body['preview'].endswith(f'/clip/{pid}/last-frame.png')
    r = client.get(f'/api/video-studio/clip/{pid}/last-frame.png')
    assert r.status_code == 200 and r.mimetype == 'image/png'
    assert r.headers['Cache-Control'] == 'no-store', 'the frame changes when the clip does'
    # The refusals, each with its own status: unknown clip, unfinished clip,
    # a staging that fails (a full disk, a read-only input folder).
    assert client.post('/api/video-studio/clip/99999/last-frame').status_code == 404
    assert client.get('/api/video-studio/clip/99999/last-frame.png').status_code == 404
    r = client.post(f'/api/video-studio/clip/{qid}/last-frame')
    assert r.status_code == 400 and 'not finished' in r.get_json()['error']

    def full(src, dest, input_dir):
        raise OSError(28, 'No space left on device')
    monkeypatch.setattr(comfy_fs, 'stage_input_image', full)
    r = client.post(f'/api/video-studio/clip/{pid}/last-frame')
    assert r.status_code == 409 and 'No space' in r.get_json()['error']


def _sounding(banner_for):
    """A fake ffmpeg: `-i` probes answer with the banner chosen per file, the
    join writes its output and records the command."""
    calls = []

    def fake(cmd, timeout=600):
        calls.append(cmd)
        if '-filter_complex' in cmd:
            with open(cmd[-1], 'wb') as fh:
                fh.write(b'J')
            return _R()
        return _R(1, banner_for(cmd[-1]))
    return fake, calls


def test_a_landed_part_is_joined_behind_its_parent_outside_the_write_transaction(app, monkeypatch, tmp_path):
    from app.extensions import db
    from app.models import VideoTestClip
    from app.services import ffmpeg_tools
    monkeypatch.setattr(vts, 'clips_dir', lambda create=True: str(tmp_path))
    monkeypatch.setattr(ffmpeg_tools, 'ffmpeg_path', lambda: 'ffmpeg')
    monkeypatch.setattr(vts, '_bring_clip_home', lambda filename: None)
    monkeypatch.setattr(vts, '_render_seconds', lambda job_id: 9.0)
    (tmp_path / 'p.mp4').write_bytes(b'P')
    (tmp_path / 'n.mp4').write_bytes(b'N')
    seen = {}
    fake, calls = _sounding(lambda path: BANNER_AUDIO)

    def watched(cmd, timeout=600):
        if '-filter_complex' in cmd:
            # The encode must not run inside a transaction: the row is
            # committed as 'done' with the part before ffmpeg starts.
            seen['in_transaction'] = db.session().in_transaction()
        return fake(cmd, timeout)
    monkeypatch.setattr(vts, '_run_ffmpeg', watched)
    with app.app_context():
        parent = VideoTestClip(status='done', filename='p.mp4', prompt='p', mode='t2v', frames=56, fps=24.0)
        db.session.add(parent)
        db.session.commit()
        _png(str(tmp_path / f'clip_{parent.id}_last.png'), 736, 416)
        part = VideoTestClip(status='pending', job_id='job-c', prompt='q', mode='i2v', frames=56, fps=24.0,
                             continues_of=parent.id)
        db.session.add(part)
        db.session.commit()
        vts.link_completed_clip('job-c', 'n.mp4', failed=False)
        row = VideoTestClip.query.get(part.id)
        assert row.status == 'done' and row.filename == 'n_joined.mp4' and row.error is None, row.error
        assert row.frames == 56 + 56 - 1, 'one frame less: the conditioning frame is dropped'
        assert row.fps == 24.0
        assert not (tmp_path / 'n.mp4').exists(), 'the part is not kept apart'
        assert seen['in_transaction'] is False, 'ffmpeg ran with a write transaction open'
        # The command the join really ran: parent then part, in THAT order,
        # scaled to the parent's last frame (736×416, not the part's), both
        # sides with sound.
        join = [c for c in calls if '-filter_complex' in c][0]
        inputs = [join[i + 1] for i, a in enumerate(join) if a == '-i']
        assert inputs == [str(tmp_path / 'p.mp4'), str(tmp_path / 'n.mp4')]
        graph = ' '.join(join)
        assert 'scale=736:416' in graph and 'fps=24' in graph
        assert '[0:a]' in graph and '[1:a]' in graph and 'concat=n=2:v=0:a=1[a]' in graph


def test_a_smoothed_parent_keeps_the_new_part_s_sound_and_one_cadence(app, monkeypatch, tmp_path):
    """A Smooth clip has no sound track and plays at 48 fps; the part rendered
    at 24 with sound. The join pads the parent's side with silence of its
    length, trims one PART frame, and counts frames at the parent's cadence."""
    from app.extensions import db
    from app.models import VideoTestClip
    from app.services import ffmpeg_tools
    monkeypatch.setattr(vts, 'clips_dir', lambda create=True: str(tmp_path))
    monkeypatch.setattr(ffmpeg_tools, 'ffmpeg_path', lambda: 'ffmpeg')
    monkeypatch.setattr(vts, '_bring_clip_home', lambda filename: None)
    monkeypatch.setattr(vts, '_render_seconds', lambda job_id: 9.0)
    (tmp_path / 's.mp4').write_bytes(b'S')
    (tmp_path / 'n.mp4').write_bytes(b'N')
    fake, calls = _sounding(lambda path: BANNER_SILENT if path.endswith('s.mp4') else BANNER_AUDIO)
    monkeypatch.setattr(vts, '_run_ffmpeg', fake)
    with app.app_context():
        parent = VideoTestClip(status='done', filename='s.mp4', prompt='p', mode='t2v', frames=109, fps=48.0, vfi_of=3)
        db.session.add(parent)
        db.session.commit()
        _png(str(tmp_path / f'clip_{parent.id}_last.png'), 736, 416)
        part = VideoTestClip(status='pending', job_id='job-s', prompt='q', mode='i2v', frames=56, fps=24.0,
                             continues_of=parent.id)
        db.session.add(part)
        db.session.commit()
        vts.link_completed_clip('job-s', 'n.mp4', failed=False)
        row = VideoTestClip.query.get(part.id)
        assert row.filename == 'n_joined.mp4' and row.error is None
        assert row.fps == 48.0 and row.frames == 109 + 55 * 2, 'the part counted at the parent cadence'
        graph = ' '.join([c for c in calls if '-filter_complex' in c][0])
        assert 'anullsrc=r=48000:cl=stereo:d=4.540[a0]' in graph, "silence of the parent's own length"
        assert '[1:a]atrim=start=0.041667' in graph, 'one PART frame (24 fps), not one parent frame'
        assert graph.count('fps=48') == 2 and '-an' not in graph


def test_a_join_that_fails_or_raises_keeps_the_part_and_says_why(app, monkeypatch, tmp_path):
    from app.extensions import db
    from app.models import VideoTestClip
    from app.services import ffmpeg_tools
    monkeypatch.setattr(vts, 'clips_dir', lambda create=True: str(tmp_path))
    monkeypatch.setattr(ffmpeg_tools, 'ffmpeg_path', lambda: 'ffmpeg')
    monkeypatch.setattr(vts, '_bring_clip_home', lambda filename: None)
    monkeypatch.setattr(vts, '_render_seconds', lambda job_id: 9.0)
    (tmp_path / 'p.mp4').write_bytes(b'P')
    with app.app_context():
        parent = VideoTestClip(status='done', filename='p.mp4', prompt='p', mode='t2v', frames=56, fps=24.0)
        db.session.add(parent)
        db.session.commit()
        pid = parent.id
        _png(str(tmp_path / f'clip_{pid}_last.png'), 736, 416)

        def land(job, name, runner):
            (tmp_path / name).write_bytes(b'X')
            monkeypatch.setattr(vts, '_run_ffmpeg', runner)
            c = VideoTestClip(status='pending', job_id=job, prompt='q', mode='i2v', frames=56, fps=24.0,
                              continues_of=pid)
            db.session.add(c)
            db.session.commit()
            vts.link_completed_clip(job, name, failed=False)
            return VideoTestClip.query.get(c.id)

        # ffmpeg refuses the graph: the part stays, the half-written joint does not.
        def refusing(cmd, timeout=600):
            if '-filter_complex' in cmd:
                with open(cmd[-1], 'wb') as fh:
                    fh.write(b'half')
                return _R(1, 'boom')
            return _R(1, BANNER_AUDIO)
        row = land('job-d', 'm.mp4', refusing)
        assert row.status == 'done' and row.filename == 'm.mp4'
        assert row.error == 'continuation not joined: boom'
        assert not (tmp_path / 'm_joined.mp4').exists(), 'no half-written joint left behind'

        # ffmpeg hangs past its timeout (an exception, not a return code): the
        # render that succeeded is not marked failed for it.
        def hanging(cmd, timeout=600):
            if '-filter_complex' in cmd:
                raise subprocess.TimeoutExpired(cmd, timeout)
            return _R(1, BANNER_AUDIO)
        row = land('job-e', 'o.mp4', hanging)
        assert row.status == 'done' and row.filename == 'o.mp4'
        assert row.error.startswith('continuation not joined:') and 'timed out' in row.error

        # No ffmpeg on this machine: said, the part kept.
        monkeypatch.setattr(ffmpeg_tools, 'ffmpeg_path', lambda: None)
        row = land('job-f', 'q.mp4', lambda cmd, timeout=600: _R())
        assert row.status == 'done' and row.filename == 'q.mp4'
        assert row.error == 'continuation not joined: ffmpeg is not available'
        monkeypatch.setattr(ffmpeg_tools, 'ffmpeg_path', lambda: 'ffmpeg')

        # The parent was deleted while the part rendered: the part stays a clip.
        db.session.delete(VideoTestClip.query.get(pid))
        db.session.commit()
        row = land('job-g', 'r.mp4', lambda cmd, timeout=600: _R())
        assert row.status == 'done' and row.filename == 'r.mp4'
        assert row.error == 'continuation not joined: the clip it continues is gone'


def test_the_history_says_joined_not_joined_or_nothing_yet(app, client, monkeypatch, tmp_path):
    from app.extensions import db
    from app.models import VideoTestClip
    monkeypatch.setattr(vts, 'clips_dir', lambda create=True: str(tmp_path))
    with app.app_context():
        parent = VideoTestClip(status='done', filename='p.mp4', prompt='p', mode='t2v')
        db.session.add(parent)
        db.session.commit()
        for i in range(30):
            db.session.add(VideoTestClip(status='done', filename=f'{i}.mp4', prompt='x', mode='t2v'))
        joined = VideoTestClip(status='done', filename='c_joined.mp4', prompt='q', mode='i2v', continues_of=parent.id)
        apart = VideoTestClip(status='done', filename='d.mp4', prompt='q', mode='i2v', continues_of=parent.id,
                              error='continuation not joined: boom')
        rendering = VideoTestClip(status='pending', prompt='q', mode='i2v', continues_of=parent.id)
        # Landed seconds ago, the join under way: done with the part, no error yet.
        joining = VideoTestClip(status='done', filename='e.mp4', prompt='q', mode='i2v', continues_of=parent.id)
        db.session.add_all([joined, apart, rendering, joining])
        db.session.commit()
        pid, ids = parent.id, (joined.id, apart.id, rendering.id, joining.id)
    body = client.get('/api/video-studio/clips?limit=6').get_json()
    rows = {c['id']: c for c in body['clips']}
    assert pid in rows, 'the parent rides along with its continuations'
    assert rows[ids[0]]['continues_of'] == pid and rows[ids[0]]['joined'] is True
    assert rows[ids[1]]['joined'] is False and rows[ids[1]]['error'].startswith('continuation not joined')
    assert rows[ids[2]]['joined'] is None, 'no verdict while the part renders'
    assert rows[ids[3]]['joined'] is None, 'no verdict while the join runs — "(not joined)" is for a failure'


def test_deleting_a_clip_drops_its_last_frame_cache_too(app, client, monkeypatch, tmp_path):
    from app.extensions import db
    from app.models import VideoTestClip
    monkeypatch.setattr(vts, 'clips_dir', lambda create=True: str(tmp_path))
    (tmp_path / 'p.mp4').write_bytes(b'P')
    with app.app_context():
        clip = VideoTestClip(status='done', filename='p.mp4', prompt='p', mode='t2v')
        db.session.add(clip)
        db.session.commit()
        cid = clip.id
    _png(str(tmp_path / f'clip_{cid}_last.png'))
    assert client.delete(f'/api/video-studio/clip/{cid}').status_code == 200
    assert not (tmp_path / 'p.mp4').exists() and not (tmp_path / f'clip_{cid}_last.png').exists()
