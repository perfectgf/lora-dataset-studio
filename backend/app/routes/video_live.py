"""🔴 Live channel API — start, stop, watch.

The stream itself is plain HLS on the app's own port: a playlist and MPEG-TS
segments served as files. That is what lets VLC on another machine of the LAN
open the same URL the browser player reads, with nothing to install and no
second server to expose — the app's network guard applies to these routes like
to every other `/api` route, so a LAN that needs the access token needs it here
too (`?token=` is accepted by the guard, which is how VLC can present it).
"""
import logging
import os

from flask import Blueprint, Response, jsonify, request, send_file

from .. import config as cfg
from .. import netguard
from ..config import LOCAL_USER
from ..services import live_studio as live
from ..services import video_test_studio as vts
from ._common import _require_comfyui, _require_no_stalled_comfyui

logger = logging.getLogger(__name__)

bp = Blueprint('video_live', __name__, url_prefix='/api/video-studio/live')

_ALLOWED_PARAMS = ('lora', 'lora_strength', 'megapixels', 'aspect', 'frames', 'fps', 'steps',
                   'turbo', 'eros', 'sparse', 'scenes', 'subject', 'seed')


def _clean_params(data):
    p = {k: data.get(k) for k in _ALLOWED_PARAMS if k in data}
    p['frames'] = vts.snap_frames(p.get('frames'))
    p['megapixels'] = vts.clamp_megapixels(p.get('megapixels', vts.MP_DEFAULT))
    fps = p.get('fps')
    try:
        fps = float(fps) if fps not in (None, '', 'auto') else 0.0
    except (TypeError, ValueError):
        fps = 0.0
    p['fps'] = max(live.FPS_MIN, min(live.FPS_MAX, fps)) if fps > 0 else 0.0
    p['turbo'] = bool(p.get('turbo', True))
    steps = p.get('steps')
    try:
        p['steps'] = int(steps) if steps not in (None, '') else None
    except (TypeError, ValueError):
        p['steps'] = None
    p['scenes'] = str(p.get('scenes') or live.DEFAULT_SCENES)
    p['subject'] = str(p.get('subject') or '')
    try:
        p['seed'] = int(p.get('seed') or 0)
    except (TypeError, ValueError):
        p['seed'] = 0
    return p


@bp.get('/options')
def live_options():
    """What the panel needs that the Studio's own options do not carry."""
    facts = live.ffmpeg_facts()
    return jsonify({
        'ffmpeg': bool(facts.get('path')), 'rubberband': bool(facts.get('rubberband')),
        'fps_min': live.FPS_MIN, 'fps_max': live.FPS_MAX, 'authored_fps': live.AUTHORED_FPS,
        'pipeline': live.PIPELINE, 'prefill': live.PREFILL, 'buffer_ahead': live.BUFFER_AHEAD,
        'default_scenes': live.DEFAULT_SCENES,
        # Other machines must present the access token (netguard's own rule):
        # the panel then tells the VLC user to put it in the address.
        'token_required': bool(netguard.public_bind() or cfg.get('server.require_token')),
    })


@bp.post('/start')
def live_start():
    """Open the channel — gated like a clip launch: ComfyUI reachable and not
    sitting on a stalled prompt."""
    blocked = _require_comfyui() or _require_no_stalled_comfyui()
    if blocked:
        return blocked
    data = request.get_json(silent=True) or {}
    lora = data.get('lora') or None
    if lora:
        from ..services import lora_test_studio as lts
        if lts._is_unsafe_external_lora_name(lora):
            return jsonify({'ok': False, 'error': 'invalid LoRA name'}), 400
    try:
        session = live.start(request_app(), LOCAL_USER, _clean_params(data))
    except live.LiveError as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 409
    return jsonify({'ok': True, **session.status()})


def request_app():
    from flask import current_app
    return current_app._get_current_object()


@bp.post('/stop')
def live_stop():
    out = live.stop()
    if out is None:
        return jsonify({'ok': True, 'state': 'idle'})
    return jsonify({'ok': True, **out})


@bp.get('/status')
def live_status():
    s = live.current()
    if s is None:
        return jsonify({'state': 'idle'})
    return jsonify(s.status())


@bp.get('/<sid>/stream.m3u8')
def live_playlist(sid):
    s = live.current()
    if s is None or s.id != sid or not os.path.isfile(s.playlist_path):
        return jsonify({'error': 'no channel'}), 404
    token = request.args.get('token')
    if token:
        # The player presented the token in the URL (VLC on a guarded LAN):
        # hand it to the segments too — a relative URI inherits no query.
        with open(s.playlist_path, encoding='utf-8') as fh:
            text = live.playlist_with_query(fh.read(), {'token': token})
        resp = Response(text, mimetype='application/vnd.apple.mpegurl')
    else:
        resp = send_file(str(s.playlist_path), mimetype='application/vnd.apple.mpegurl',
                         conditional=False, max_age=0)
    resp.headers['Cache-Control'] = 'no-store'
    return resp


@bp.get(f'/<sid>/{live.SEGMENT_DIR}/<name>')
def live_segment(sid, name):
    s = live.current()
    if s is None or s.id != sid or not live.is_segment_name(name):
        return jsonify({'error': 'no such segment'}), 404
    path = s.dir / name
    if not path.is_file():
        return jsonify({'error': 'no such segment'}), 404
    s.note_segment_request(name)
    resp = send_file(str(path), mimetype='video/mp2t', conditional=True, max_age=0)
    resp.headers['Cache-Control'] = 'no-store'
    return resp
