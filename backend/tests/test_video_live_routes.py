"""🔴 The live channel's routes: gated like a launch, served like files."""
from urllib.parse import urljoin

import pytest

from app.services import live_studio as live


class _FakeSession:
    def __init__(self, tmp_path, sid='abcd1234'):
        self.id = sid
        self.dir = tmp_path / 'live' / sid
        self.dir.mkdir(parents=True)
        self.playlist_path = self.dir / 'stream.m3u8'
        self.requested = []
        self.state = 'running'

    def status(self):
        return {'id': self.id, 'state': self.state, 'produced': 3}

    def note_segment_request(self, name):
        self.requested.append(name)


@pytest.fixture
def fake_session(tmp_path, monkeypatch):
    s = _FakeSession(tmp_path)
    monkeypatch.setattr(live, 'current', lambda: s)
    return s


def test_options_say_what_ffmpeg_can_do_and_the_rates_the_stream_accepts(client, monkeypatch):
    monkeypatch.setattr(live, 'ffmpeg_facts', lambda force=False: {'path': 'ffmpeg', 'rubberband': False})
    body = client.get('/api/video-studio/live/options').get_json()
    assert body['ffmpeg'] is True and body['rubberband'] is False
    assert body['fps_min'] == live.FPS_MIN and body['fps_max'] == live.FPS_MAX
    assert body['default_scenes'] == live.DEFAULT_SCENES
    assert body['token_required'] is False, 'a home LAN is trusted by default'


def test_options_say_when_vlc_needs_the_token_in_the_address(client, monkeypatch):
    from app.routes import video_live
    monkeypatch.setattr(live, 'ffmpeg_facts', lambda force=False: {'path': 'ffmpeg', 'rubberband': True})
    monkeypatch.setattr(video_live.cfg, 'get', lambda key, default=None: True if key == 'server.require_token' else default)
    assert client.get('/api/video-studio/live/options').get_json()['token_required'] is True
    monkeypatch.setattr(video_live.cfg, 'get', lambda key, default=None: default)
    monkeypatch.setattr(video_live.netguard, 'public_bind', lambda: True)
    assert client.get('/api/video-studio/live/options').get_json()['token_required'] is True


def test_start_is_gated_on_comfyui_like_a_clip_launch(client, monkeypatch):
    from flask import jsonify
    from app.routes import video_live
    monkeypatch.setattr(video_live, '_require_comfyui',
                        lambda **k: (jsonify({'error': 'ComfyUI is not reachable'}), 409))
    r = client.post('/api/video-studio/live/start', json={'scenes': 'x'})
    assert r.status_code == 409


def test_start_cleans_its_parameters_and_answers_the_channels_status(client, monkeypatch, tmp_path):
    from app.routes import video_live
    monkeypatch.setattr(video_live, '_require_comfyui', lambda **k: None)
    monkeypatch.setattr(video_live, '_require_no_stalled_comfyui', lambda: None)
    seen = {}

    def fake_start(app, user_id, params):
        seen.update(params)
        return _FakeSession(tmp_path)
    monkeypatch.setattr(live, 'start', fake_start)
    r = client.post('/api/video-studio/live/start', json={
        'frames': 130, 'megapixels': 9, 'fps': '99', 'steps': '4', 'scenes': '',
        'subject': 'Jessy', 'seed': 'x', 'lora': 'h3/lds/mine.safetensors', 'ignored': 1})
    assert r.status_code == 200 and r.get_json()['produced'] == 3
    assert seen['frames'] == 124                      # snapped to the VAE grid
    assert seen['megapixels'] == 2.0                  # clamped
    assert seen['fps'] == live.FPS_MAX                # clamped
    assert seen['steps'] == 4 and seen['seed'] == 0
    assert seen['scenes'] == live.DEFAULT_SCENES      # empty → the shipped scenes
    assert seen['lora'] == 'h3/lds/mine.safetensors' and 'ignored' not in seen


def test_start_refuses_a_rooted_lora_name_and_a_second_channel(client, monkeypatch):
    from app.routes import video_live
    monkeypatch.setattr(video_live, '_require_comfyui', lambda **k: None)
    monkeypatch.setattr(video_live, '_require_no_stalled_comfyui', lambda: None)
    r = client.post('/api/video-studio/live/start', json={'lora': '../../etc/passwd'})
    assert r.status_code == 400

    def refuse(app, user_id, params):
        raise live.LiveError('A channel is already running — stop it first.')
    monkeypatch.setattr(live, 'start', refuse)
    r = client.post('/api/video-studio/live/start', json={})
    assert r.status_code == 409 and 'already running' in r.get_json()['error']


def test_status_and_stop_are_idle_without_a_channel(client, monkeypatch):
    monkeypatch.setattr(live, 'current', lambda: None)
    monkeypatch.setattr(live, 'stop', lambda: None)
    assert client.get('/api/video-studio/live/status').get_json() == {'state': 'idle'}
    assert client.post('/api/video-studio/live/stop').get_json() == {'ok': True, 'state': 'idle'}
    assert client.get('/api/video-studio/live/abcd1234/stream.m3u8').status_code == 404


def test_the_playlist_and_segments_are_served_uncached_and_the_segment_is_noted(client, fake_session):
    fake_session.playlist_path.write_text('#EXTM3U\n', encoding='utf-8')
    (fake_session.dir / 'seg_000002.ts').write_bytes(b'TS')
    r = client.get('/api/video-studio/live/abcd1234/stream.m3u8')
    assert r.status_code == 200
    assert r.mimetype == 'application/vnd.apple.mpegurl'
    assert r.headers['Cache-Control'] == 'no-store'
    r = client.get('/api/video-studio/live/abcd1234/seg/seg_000002.ts')
    assert r.status_code == 200 and r.mimetype == 'video/mp2t' and r.data == b'TS'
    assert fake_session.requested == ['seg_000002.ts']
    # Another channel id, a missing segment, a name of the wrong shape: all 404.
    assert client.get('/api/video-studio/live/zzzz/stream.m3u8').status_code == 404
    assert client.get('/api/video-studio/live/abcd1234/seg/seg_000009.ts').status_code == 404
    assert client.get('/api/video-studio/live/abcd1234/seg/stream.m3u8').status_code == 404
    assert client.get('/api/video-studio/live/abcd1234/seg/..%2Fstream.m3u8').status_code == 404
    assert fake_session.requested == ['seg_000002.ts'], 'only a served segment moves the viewer'


def test_the_playlist_names_its_segments_where_a_player_resolves_them(client, fake_session):
    """ffmpeg, hls.js and VLC all resolve a relative URI against the playlist's
    URL. The smoke run against the real engine found every player on a 404
    because the playlist listed bare file names — this walks the same path."""
    fake_session.playlist_path.write_text(live.playlist_text([(2, 'seg_000002.ts', 9.333)]), encoding='utf-8')
    (fake_session.dir / 'seg_000002.ts').write_bytes(b'TS')
    playlist_url = '/api/video-studio/live/abcd1234/stream.m3u8'
    body = client.get(playlist_url).get_data(as_text=True)
    uri = [l for l in body.splitlines() if l and not l.startswith('#')][0]
    r = client.get(urljoin(playlist_url, uri))
    assert r.status_code == 200 and r.data == b'TS'
    # With the access token in the playlist's query (VLC on a guarded LAN), the
    # segments carry it too — and the plain playlist stays clean.
    body = client.get(playlist_url + '?token=abc').get_data(as_text=True)
    uri = [l for l in body.splitlines() if l and not l.startswith('#')][0]
    assert uri == 'seg/seg_000002.ts?token=abc'
    assert client.get(urljoin(playlist_url, uri)).status_code == 200
    assert 'token' not in client.get(playlist_url).get_data(as_text=True)
