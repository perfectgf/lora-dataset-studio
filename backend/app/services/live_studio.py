"""🔴 Live — a continuous text-to-video channel, from the Studio's own graph.

WHAT IT IS
The Video Test Studio renders one clip per launch and compares them in time.
This lane is the other shape of the same engine: it never stops. A scene is
drawn from a list, a clip is rendered, the next scene is already in the queue,
and every finished clip is appended to a stream that a player reads like a TV
channel — in the browser, or in VLC on any machine of the LAN.

The shape comes from FastH3 Live (jacokon, Apache-2.0 — "An endless AI TV
channel on a single gaming GPU", r/StableDiffusion, 2026-09), whose write-up is
the source of two ideas this module keeps and one it drops:

* KEPT — generation is slower than 24 fps playback on one card, so the stream
  is RETIMED on the way out: the frames are authored at 24 fps and played at a
  lower rate, the audio slowed by the same factor with its pitch preserved.
  The break-even rate is `frames / seconds per clip`, measured on a trailing
  window of real clips, never planned from a best case.
* KEPT — the producer keeps two prompts in the queue, so the card never idles
  between a job finishing and the next one being accepted.
* DROPPED — its 4-step FastH3 student and its custom writer node. The
  maintainer judged this app's turbo LoRA at 4 steps the better picture
  (bench of 2026-09-03), it is already installed, and the graph stays the
  Studio's own: nothing new to download, no node pack.

HOW IT IS BUILT
* The PRODUCER thread draws a scene, builds the Studio graph
  (`video_test_studio.build_workflow`, t2v) and hands it to the app's own
  queue with the `is_live` marker. It tops the queue up to `pipeline` jobs and
  pauses when the viewer is far enough behind: a segment request tells the
  server where the player is (`note_segment_request`), and nothing is rendered
  more than `buffer_ahead` clips past it. No viewer yet = a bounded prefill.
* The queue worker completes the job like any other and `_dispatch_completion`
  routes it here (`link_completed_live_clip`): the mp4 is claimed out of
  ComfyUI's output folder and handed to the FEEDER thread with the seconds the
  queue spent on it.
* The FEEDER retimes each clip with ffmpeg into one MPEG-TS segment at the
  playback rate, appends it to a sliding HLS playlist, and deletes the source.
  HLS rather than the write-up's own MPEG-TS pusher: VLC and hls.js both read
  a playlist natively, the player paces itself (no `-re`), and a viewer can
  join, leave and rejoin without the muxer ever noticing.
* The playback rate is the user's, or AUTO: decided after the prefill from the
  measured clips, a tenth under what the card sustains so the buffer grows
  instead of draining. The status line says what the card sustains right now
  and whether the stream is keeping up, with the same arithmetic the write-up
  prints per clip.

One session at a time: the lane owns the card while it runs, and a second
channel would only halve the first.

Live clips are EPHEMERAL: no `video_test_clip` row, nothing in the history —
a channel that produced a thousand clips must not leave a thousand cards.
"""
from __future__ import annotations

import collections
import logging
import math
import os
import queue
import random
import re
import shutil
import subprocess
import threading
import urllib.parse
import time
import uuid

from ..utils.redact import redact_user_paths

logger = logging.getLogger(__name__)

# The job name stamped on every live clip. Routed by `is_live` in
# `_dispatch_completion` (checked before model_name), so the harvest contract
# lists it as a non-dataset job: its result comes home through
# `link_completed_live_clip`, never through a dataset row.
JOB_NAME = 'video_live'

PIPELINE = 2          # prompts kept in the queue (the write-up's --pipeline)
BUFFER_AHEAD = 4      # clips rendered past the viewer before the producer pauses
PREFILL = 2           # clips banked before the playback rate is decided
WINDOW = 8            # clips averaged for the sustainable-fps readout
FPS_MIN, FPS_MAX = 6.0, 24.0
SUBMIT_RETRIES = 5                          # refused submits in a row before the channel closes itself
SUBMIT_BACKOFF = (1.5, 3.0, 6.0, 12.0, 24.0)  # seconds between them
AUTHORED_FPS = 24.0   # H3 authors motion at 24 fps; retiming plays it slower
AUTO_HEADROOM = 0.9   # auto rate = a tenth under what the card sustains
SEGMENT_KEEP = 2      # segments kept behind the viewer, for a rewind or a retry
SCENE_SEPARATOR = '---'
NAME_SLOT = re.compile(r'\{NAME\d*\}')
SEGMENT_NAME = re.compile(r'^seg_\d{6}\.ts$')

DEFAULT_SCENES = """integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium shot of {NAME} at a wooden kitchen table by a window in soft morning light, lifting a mug with both hands, taking a slow sip and looking out of the window with a small smile. The camera holds still.

overall_soundscape: Quiet room tone, a distant kettle, a soft clink of the mug on the table.

non_diegetic_music: N/A
---
integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium shot of {NAME} walking toward the camera along a sunlit corridor, hair swaying with each step, reaching a glass door and pushing it open. The camera tracks backward slowly at medium distance.

overall_soundscape: Soft footsteps on a hard floor, the hush of an office, a glass door swinging open with a faint creak.

non_diegetic_music: N/A
---
integrated_multimodal_description: [Shot 1] Live-action, cinematic, a close shot of {NAME} on a balcony at dusk, city lights out of focus behind, turning toward the camera and tucking a strand of hair behind one ear. The camera pushes in with small amplitude at slow speed.

overall_soundscape: Distant traffic, a light breeze, the faint hum of the city.

non_diegetic_music: N/A"""


class LiveError(ValueError):
    """A refusal with a sentence the UI can show as-is."""


# ── Pure helpers (tested without a GPU, a queue or ffmpeg) ──────────────────

def parse_scenes(text) -> list:
    """Scene blocks separated by a line holding `---`; blank blocks dropped."""
    out = []
    for block in re.split(r'(?m)^\s*' + re.escape(SCENE_SEPARATOR) + r'\s*$', str(text or '')):
        block = block.strip()
        if block:
            out.append(block)
    return out


def fill_scene(scene, subject) -> str:
    """Every `{NAME}` / `{NAME2}` slot becomes the subject — the trigger word
    of the LoRA under test, or whatever the user typed."""
    who = (subject or '').strip() or 'a person'
    return NAME_SLOT.sub(who, scene)


def sustain_fps(frames, render_seconds) -> float | None:
    """The playback rate this render pace keeps up with: frames per second of
    GPU time. None until there is a measurement."""
    try:
        f, s = float(frames), float(render_seconds)
    except (TypeError, ValueError):
        return None
    if f <= 0 or s <= 0:
        return None
    return f / s


def auto_fps(frames, mean_render_seconds) -> float:
    """The rate AUTO picks: a tenth under what the card sustains, clamped to
    the range the stream can retime to, rounded down to a whole frame."""
    sustain = sustain_fps(frames, mean_render_seconds)
    if sustain is None:
        return FPS_MIN
    return float(max(FPS_MIN, min(FPS_MAX, math.floor(sustain * AUTO_HEADROOM))))


def verdict(frames, play_fps, mean_render_seconds, buffered_clips) -> dict:
    """The write-up's per-clip line, as numbers: seconds of playback a clip
    buys, seconds of render it costs, the margin, and how long the buffer
    lasts when the margin is negative. Keyed `pace`, never `state`: the
    session's own `state` (starting/running/stopped) sits beside it."""
    play_seconds = float(frames) / float(play_fps) if play_fps else None
    out = {'play_seconds': round(play_seconds, 2) if play_seconds else None,
           'render_seconds': round(mean_render_seconds, 1) if mean_render_seconds else None,
           'sustain_fps': None, 'margin_seconds': None, 'pace': 'measuring',
           'runway_clips': None}
    if not mean_render_seconds or not play_seconds:
        return out
    out['sustain_fps'] = round(sustain_fps(frames, mean_render_seconds), 1)
    margin = play_seconds - mean_render_seconds
    out['margin_seconds'] = round(margin, 2)
    if margin >= 0:
        out['pace'] = 'keeping_up'
    else:
        out['pace'] = 'behind'
        if buffered_clips:
            out['runway_clips'] = int(buffered_clips * play_seconds / -margin)
    return out


def audio_stretch(rate, rubberband_available) -> str:
    """The filter that slows the audio to `rate` without changing its pitch.

    rubberband is a real phase vocoder with no floor and is used when the
    ffmpeg build carries it (imageio's does). atempo bottoms out at 0.5 per
    stage and rings on broadband sound, so under it the slow-down is split
    into stages that each stay above the floor.
    """
    rate = float(rate)
    if rubberband_available:
        return f'rubberband=tempo={rate:.8f}:transients=smooth'
    stages = max(1, math.ceil(math.log(rate) / math.log(0.5))) if rate < 1 else 1
    per = rate ** (1.0 / stages)
    return ','.join(f'atempo={per:.8f}' for _ in range(stages))


def retime_command(ffmpeg, src, dst, play_fps, offset_seconds, rubberband_available,
                   with_audio=True) -> list:
    """The ffmpeg call that turns one 24 fps clip into one MPEG-TS segment at
    the playback rate, timestamped to follow the previous segment.

    Video: `setpts` stretches the timeline, x264 at a fixed GOP with no scene
    cut so every segment starts on a keyframe. Audio: stretched by the same
    factor, AAC at a normal rate. Same PIDs on every segment, so a boundary is
    not a programme change for the player.
    """
    rate = float(play_fps) / AUTHORED_FPS
    gop = str(int(round(float(play_fps))))
    if with_audio:
        graph = (f'[0:v]setpts=PTS/{rate:.6f}[v];'
                 f'[0:a]{audio_stretch(rate, rubberband_available)},aresample=48000[a]')
        maps = ['-map', '[v]', '-map', '[a]']
        audio = ['-c:a', 'aac', '-b:a', '128k', '-ar', '48000', '-ac', '2']
    else:
        graph = f'[0:v]setpts=PTS/{rate:.6f}[v]'
        maps = ['-map', '[v]']
        audio = ['-an']
    return [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-i', src,
            '-filter_complex', graph, *maps, '-r', f'{float(play_fps):g}',
            '-c:v', 'libx264', '-preset', 'veryfast', '-tune', 'zerolatency',
            '-g', gop, '-keyint_min', gop, '-sc_threshold', '0', '-pix_fmt', 'yuv420p',
            '-b:v', '2M', *audio, '-muxdelay', '0',
            '-output_ts_offset', f'{float(offset_seconds):.3f}',
            '-streamid', '0:256', '-streamid', '1:257',
            '-mpegts_flags', '+resend_headers', '-pat_period', '0.1', '-sdt_period', '0.5',
            '-f', 'mpegts', dst]


SEGMENT_DIR = 'seg'   # the path element between the playlist and its segments


def playlist_text(segments, ended=False, discontinuities=()) -> str:
    """A live HLS playlist over `segments` = [(seq, name, seconds), …] in order.
    `discontinuities`: the sequence numbers before which the elementary streams
    change (a segment without audio after ones with it) — tagged, so a player
    resets its decoders instead of stalling.

    `EXT-X-MEDIA-SEQUENCE` is the first sequence number listed, which is how a
    player follows a window that slides; `EXT-X-ENDLIST` only when the channel
    has stopped, so a player drains what is left and does not wait for more.

    Segment URIs are RELATIVE, and a player resolves them against the
    playlist's own URL: `seg/seg_000007.ts` next to `.../stream.m3u8` is
    `.../seg/seg_000007.ts`, where the route serves them. A bare file name
    here resolved one level too high and every player got a 404 — found by
    reading the stream with ffmpeg, not by a unit test.
    """
    segs = list(segments)
    target = max([math.ceil(float(s[2])) for s in segs] + [1])
    lines = ['#EXTM3U', '#EXT-X-VERSION:3', f'#EXT-X-TARGETDURATION:{target}',
             f'#EXT-X-MEDIA-SEQUENCE:{segs[0][0] if segs else 0}']
    for seq, name, seconds in segs:
        if seq in discontinuities:
            lines.append('#EXT-X-DISCONTINUITY')
        lines.append(f'#EXTINF:{float(seconds):.3f},')
        lines.append(f'{SEGMENT_DIR}/{name}')
    if ended:
        lines.append('#EXT-X-ENDLIST')
    return '\n'.join(lines) + '\n'


def playlist_with_query(text, params) -> str:
    """The same playlist with `params` appended to every segment URI.

    A player that presented the access token in the playlist's URL (VLC on a
    network where the app requires one) fetches the segments with the same
    query, so they get through the guard the way the playlist did — a relative
    URI inherits the path, never the query.
    """
    query = urllib.parse.urlencode(params)
    if not query:
        return text
    out = []
    for line in text.splitlines():
        if line and not line.startswith('#'):
            line = f'{line}?{query}'
        out.append(line)
    return '\n'.join(out) + '\n'


def _run_ffmpeg(cmd):
    """The app's subprocess convention for ffmpeg (video_bank_service has the
    same): no console window in the frozen Windows build, utf-8 stderr with
    replacement, and a timeout — a clip is seconds, never minutes."""
    return subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace',
                          creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), timeout=600)


def segment_name(seq) -> str:
    return f'seg_{int(seq):06d}.ts'


def is_segment_name(name) -> bool:
    """Only the exact shape this module writes: a route must never build a
    path out of a client string that is anything else."""
    return bool(SEGMENT_NAME.match(str(name or '')))


# ── ffmpeg facts, read once ─────────────────────────────────────────────────

_FFMPEG_FACTS = {}


def ffmpeg_facts(force=False) -> dict:
    """{'path': …, 'rubberband': bool} — what the ffmpeg on this machine can do."""
    if _FFMPEG_FACTS and not force:
        return _FFMPEG_FACTS
    from . import ffmpeg_tools
    path = ffmpeg_tools.ffmpeg_path()
    rubber = False
    if path:
        try:
            r = subprocess.run([path, '-hide_banner', '-filters'], capture_output=True,
                               text=True, timeout=20)
            rubber = bool(re.search(r'^\s*\S+\s+rubberband\s', r.stdout or '', re.M))
        except (OSError, subprocess.SubprocessError):
            rubber = False
    _FFMPEG_FACTS.clear()
    _FFMPEG_FACTS.update({'path': path, 'rubberband': rubber})
    return _FFMPEG_FACTS


# ── The session ─────────────────────────────────────────────────────────────

def live_root(create=False):
    from .. import config as cfg
    root = cfg.data_dir() / 'live'
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


class LiveSession:
    """One running channel: its parameters, its threads, its stream on disk."""

    def __init__(self, app, user_id, params):
        self.app = app
        self.user_id = user_id
        self.id = uuid.uuid4().hex[:8]
        self.params = params
        self.scenes = parse_scenes(params.get('scenes'))
        if not self.scenes:
            raise LiveError('Write at least one scene — the channel has nothing to show.')
        self.dir = live_root(create=True) / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.playlist_path = self.dir / 'stream.m3u8'
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.state = 'starting'
        self.error = None
        self.created_at = time.time()
        # Producer bookkeeping.
        self.submitted = 0
        self.inflight = {}            # job_id -> seq
        self.abandoned = set()        # left on the card at stop: discarded when they land
        self.scene_cursor = list(range(len(self.scenes)))
        random.shuffle(self.scene_cursor)
        self.last_prompt = ''
        self.last_scene_index = None
        # Feeder bookkeeping.
        self.pending = queue.Queue()  # (seq, path, render_seconds), in completion order
        self.segments = []            # [(media sequence, name, seconds)] on disk — contiguous, HLS's own numbering
        self.discontinuities = set()  # media sequences where the elementary streams change (a silent clip)
        self.last_silent = False
        self.offset = 0.0             # playback seconds already streamed
        self.produced = 0
        self.failed = 0
        self.render_times = collections.OrderedDict()   # seq -> seconds, at completion
        self.play_fps = float(params.get('fps') or 0) or None   # None = auto, undecided
        self.auto = self.play_fps is None
        self.last_requested_seq = 0
        self.threads = []

    # -- what the panel reads ------------------------------------------------

    def status(self) -> dict:
        with self.lock:
            mean = self._mean_render_locked()
            latest = self.segments[-1][0] if self.segments else 0
            buffered = max(0, latest - self.last_requested_seq)
            frames = int(self.params.get('frames') or 0)
            v = verdict(frames, self.play_fps, mean, buffered) if frames else {}
            # The producer stops at BUFFER_AHEAD clips nobody has asked for:
            # a channel nobody watches costs nothing, and the rail says so.
            paused = (self.state in ('starting', 'running')
                      and self._ahead_locked() >= BUFFER_AHEAD + PIPELINE)
            return {
                'id': self.id, 'state': self.state,
                'error': redact_user_paths(self.error) if self.error else None,
                'paused_for_viewer': paused,
                'measured': sum(1 for t in self.render_times.values() if t and t > 0),
                'params': dict(self.params),
                'play_fps': self.play_fps, 'auto_fps': self.auto,
                'produced': self.produced, 'failed': self.failed,
                'inflight': len(self.inflight), 'segments': len(self.segments),
                'latest_seq': latest, 'viewer_seq': self.last_requested_seq,
                'buffered_clips': buffered,
                'last_prompt': self.last_prompt, 'last_scene_index': self.last_scene_index,
                'scene_count': len(self.scenes),
                'playlist': f'/api/video-studio/live/{self.id}/stream.m3u8',
                **v,
            }

    def _mean_render_locked(self):
        """Mean of the last WINDOW measured clips — minus the first one that
        came back, once there is anything else: it carries the model load, not
        the pace. The FIRST MEASURED clip, not clip #1: when clip #1 is refused
        at submit (seen on the real engine, right after a start) the cold one
        is #2, and keying on the number would halve the rate for good."""
        times = sorted((seq, t) for seq, t in self.render_times.items() if t and t > 0)
        if len(times) > 1:
            times = times[1:]
        values = [t for _seq, t in times[-WINDOW:]]
        return (sum(values) / len(values)) if values else None

    def _ahead_locked(self) -> int:
        """Clips rendered, rendering or waiting beyond where the viewer is."""
        latest = self.segments[-1][0] if self.segments else 0
        return latest + len(self.inflight) + self.pending.qsize() - self.last_requested_seq

    # -- producer ------------------------------------------------------------

    def _next_scene(self):
        if not self.scene_cursor:
            self.scene_cursor = list(range(len(self.scenes)))
            random.shuffle(self.scene_cursor)
        idx = self.scene_cursor.pop()
        return idx, fill_scene(self.scenes[idx], self.params.get('subject'))

    def _may_submit(self) -> bool:
        with self.lock:
            if len(self.inflight) >= PIPELINE:
                return False
            return self._ahead_locked() < BUFFER_AHEAD + PIPELINE

    def _submit_one(self):
        from ..job_queue import queue_manager
        from . import video_test_studio as vts
        idx, prompt = self._next_scene()
        seq = self.submitted + 1
        seed = (int(self.params.get('seed') or 0) + self.submitted) & 0x7FFFFFFF
        p = self.params
        classes = vts.registered_classes()
        built = vts.build_workflow(
            prompt=prompt, mode='t2v', aspect=p.get('aspect', 'landscape'),
            megapixels=p.get('megapixels', vts.MP_DEFAULT), frames=p.get('frames'),
            seed=seed, steps=p.get('steps'), lora=p.get('lora') or None,
            lora_strength=p.get('lora_strength', 1.0), turbo=bool(p.get('turbo', True)),
            eros=bool(p.get('eros')), eros_on_disk=vts.eros_on_disk() if p.get('eros') else False,
            sparse=p.get('sparse', ''), sage=vts.sage_available(classes),
            filename_prefix=f'{vts.new_prefix(self.user_id)}_live_{self.id}')
        job_id = queue_manager.add_job(
            job_type='image', user_id=str(self.user_id), workflow_data=built['workflow'],
            prompt=prompt,
            # A literal, on purpose: test_dataset_job_harvest discovers every
            # stamped model_name by AST and a constant is invisible to it.
            metadata={'model_name': 'video_live', 'is_live': True, 'live_session': self.id,
                      'live_seq': seq, 'seed': built['seed']})
        with self.lock:
            self.submitted += 1
            self.inflight[job_id] = seq
            self.last_prompt = prompt
            self.last_scene_index = idx
            late = self.stop_event.is_set()
        if late:
            # stop() ran between the check above and this registration and did
            # not see the job: give it back here, or it renders for nobody.
            with self.lock:
                self.inflight.pop(job_id, None)
            try:
                queue_manager.cancel_job(job_id, self.user_id)
            except Exception:  # noqa: BLE001 — a cancel that fails is a clip that lands late and is ignored
                logger.debug('live %s: late cancel of %s failed', self.id, job_id, exc_info=True)
            return
        logger.info('live %s: queued clip #%d (scene %d, seed %d)', self.id, seq, idx, built['seed'])

    def _produce(self):
        refused = 0
        with self.app.app_context():
            while not self.stop_event.is_set():
                try:
                    submitted = False
                    while self._may_submit() and not self.stop_event.is_set():
                        self._submit_one()
                        submitted = True
                        refused = 0
                    if self.state == 'starting':
                        self.state = 'running'
                except Exception as exc:  # noqa: BLE001 — the loop must survive one bad submit
                    # One stack for the first refusal, a line for the next ones:
                    # a channel left running against a stopped ComfyUI wrote a
                    # stack every 1.5 s and rotated the whole log away in hours.
                    refused += 1
                    if refused == 1:
                        logger.exception('live %s: producer error', self.id)
                    else:
                        logger.warning('live %s: submit refused again (%d/%d): %s',
                                       self.id, refused, SUBMIT_RETRIES, str(exc)[:200])
                    self.error = str(exc)[:300]
                    submitted = False
                    if refused >= SUBMIT_RETRIES:
                        # Not a hiccup: close the channel cleanly (the feeder
                        # writes ENDLIST) rather than retry until someone notices.
                        self.error = f'channel stopped, ComfyUI refused {refused} submits in a row: {exc}'[:300]
                        logger.warning('live %s: stopping after %d refused submits', self.id, refused)
                        self.stop()
                        break
                    self.stop_event.wait(SUBMIT_BACKOFF[min(refused, len(SUBMIT_BACKOFF)) - 1])
                    continue
                self.stop_event.wait(0.5 if submitted else 1.5)

    # -- completion (queue monitor thread) -----------------------------------

    def on_completed(self, job_id, filename, failed, reason, render_seconds):
        with self.lock:
            seq = self.inflight.pop(job_id, None)
            abandoned = job_id in self.abandoned
            self.abandoned.discard(job_id)
        if seq is None:
            if abandoned and filename and not failed:
                self._discard_output(filename)
            else:
                logger.info('live %s: completion for an unknown job %s ignored', self.id, job_id)
            return
        if failed or not filename:
            with self.lock:
                self.failed += 1
                self.error = reason or 'a clip failed to render'
            logger.warning('live %s: clip #%d failed: %s', self.id, seq, reason)
            return
        src = self._claim(filename, seq)
        if src is None:
            with self.lock:
                self.failed += 1
            return
        with self.lock:
            if render_seconds and render_seconds > 0:
                self.render_times[seq] = float(render_seconds)
            # A clip that came back whole is the proof the pipeline works again:
            # the last error (a refused submit, a clip that failed) is over. The
            # failed count keeps the history.
            self.error = None
        self.pending.put((seq, src, render_seconds))

    def _discard_output(self, filename):
        """A clip that finished after the stop rendered for nobody: out of
        ComfyUI's output folder, when that folder is on this machine."""
        from . import lora_test_studio as lts
        out_dir = lts._comfy_output_dir()
        if not out_dir:
            return
        path = os.path.join(out_dir, filename)
        try:
            if os.path.isfile(path):
                os.remove(path)
                logger.info('live %s: discarded the clip that landed after the stop (%s)', self.id, filename)
        except OSError:
            logger.debug('live %s: could not discard %s', self.id, filename, exc_info=True)

    def _job_status(self, job_id):
        """The queue row's status, or None when the queue does not know the job."""
        from ..models import ImageGenerationQueue
        try:
            row = ImageGenerationQueue.query.filter_by(job_id=str(job_id)).first()
        except Exception:  # noqa: BLE001 — a session error reads as "unknown", never as "pending"
            return None
        return row.status if row is not None else None

    def _claim(self, filename, seq):
        """The mp4 out of ComfyUI's folder into the session's — same claim as
        the Studio's `_bring_clip_home`, disk first, `/view` as the fallback."""
        from . import lora_test_studio as lts
        from ..utils import comfy_fs
        dst = str(self.dir / f'clip_{seq:06d}.mp4')
        out_dir = lts._comfy_output_dir()
        src = os.path.join(out_dir, filename) if out_dir else None
        try:
            if src and comfy_fs.claim_output_file(src, dst):
                return dst
            from ..utils.comfyui import fetch_output_image_bytes
            data = fetch_output_image_bytes(filename)
            if data:
                with open(dst, 'wb') as fh:
                    fh.write(data)
                return dst
        except Exception:  # noqa: BLE001 — a lost clip is one clip, not the channel
            logger.exception('live %s: could not claim %s', self.id, filename)
        return None

    # -- feeder --------------------------------------------------------------

    def _decide_fps(self, final=False):
        """AUTO: after the prefill, a tenth under what the card sustains. Once.

        `final` — the channel is closing with a clip still waiting for the
        rate: decide from what was measured (one clip, or nothing → the floor)
        so the clip plays and the playlist ends, instead of the feeder waiting
        for a prefill that will never complete. Without this a stop in AUTO
        before the second clip looped the feeder forever, the state never
        reached `stopped` and the next start was refused for the life of the
        process (found by the refuter, with a witness)."""
        if self.play_fps is not None:
            return
        measured = [t for t in self.render_times.values() if t and t > 0]
        if len(measured) >= PREFILL or (final and measured):
            mean = self._mean_render_locked()
            self.play_fps = auto_fps(int(self.params.get('frames') or 0), mean)
            logger.info('live %s: playback rate decided at %.0f fps (mean render %.1fs)',
                        self.id, self.play_fps, mean)
        elif final:
            self.play_fps = FPS_MIN
            logger.info('live %s: closing with nothing measured, the last clip plays at %.0f fps',
                        self.id, self.play_fps)

    def _feed(self):
        facts = ffmpeg_facts()
        while not (self.stop_event.is_set() and self.pending.empty()):
            try:
                seq, path, render_seconds = self.pending.get(timeout=0.5)
            except queue.Empty:
                continue
            consumed = True
            try:
                consumed = self._encode(seq, path, render_seconds, facts)
            except Exception as exc:  # noqa: BLE001 — one bad clip must not end the channel
                logger.exception('live %s: could not encode clip #%d', self.id, seq)
                with self.lock:
                    self.failed += 1
                    self.error = f'encode failed: {exc}'[:300]
            if consumed:
                try:
                    os.remove(path)
                except OSError:
                    pass
            else:
                # Still prefilling: the clip went back to the queue unencoded,
                # its file untouched. Brief — the prefill is two clips.
                self.stop_event.wait(0.5)
        try:
            self._write_playlist(ended=True)
        except Exception:  # noqa: BLE001 — the channel must reach 'stopped' whatever the disk says
            logger.exception('live %s: could not end the playlist', self.id)
        self.state = 'stopped'

    def _encode(self, seq, path, render_seconds, facts) -> bool:
        """One clip into one segment. False when the clip was put back to wait
        for the playback rate (AUTO, still prefilling); True once consumed."""
        with self.lock:
            self._decide_fps(final=self.stop_event.is_set())
            fps = self.play_fps
        if fps is None:
            self.pending.put((seq, path, render_seconds))
            return False
        frames = int(self.params.get('frames') or 0)
        seconds = frames / fps
        with self.lock:
            offset = self.offset
            # HLS numbers SEGMENTS, contiguously: a clip that failed to render
            # must not leave a hole, or a player reloading the playlist across
            # it counts one segment off and replays or skips one.
            media = self.produced + 1
        name = segment_name(media)
        dst = str(self.dir / name)
        tmp = dst + '.tmp'   # a player never sees a half-written segment
        cmd = retime_command(facts['path'], path, tmp, fps, offset, facts['rubberband'])
        r = _run_ffmpeg(cmd)
        silent = False
        if r.returncode != 0:
            # A clip without an audio track (the graph always makes one, but the
            # stream must not die on the day it does not): retry video-only.
            cmd = retime_command(facts['path'], path, tmp, fps, offset, facts['rubberband'],
                                 with_audio=False)
            r = _run_ffmpeg(cmd)
            if r.returncode != 0:
                raise RuntimeError(redact_user_paths((r.stderr or '')[-400:]) or f'ffmpeg exited {r.returncode}')
            silent = True
        os.replace(tmp, dst)
        with self.lock:
            if silent != self.last_silent:
                # The elementary streams change here (audio gone, or back):
                # HLS tells the player with a discontinuity tag.
                self.discontinuities.add(media)
                self.last_silent = silent
            self.segments.append((media, name, seconds))
            self.offset += seconds
            self.produced += 1
            self._prune_locked()
        self._write_playlist()
        return True

    def _prune_locked(self):
        """Drop segments the viewer is done with.

        "The viewer" is the most advanced player: a second one further back
        (VLC opened after the tab) may find a segment gone and resync at the
        live edge — hls.js does, measured: four 404s, one stall, playing again.
        """
        keep_from = max(1, self.last_requested_seq - SEGMENT_KEEP)
        keep, drop = [], []
        for s in self.segments:
            (keep if s[0] >= keep_from else drop).append(s)
        self.segments = keep
        for seq, name, _sec in drop:
            self.discontinuities.discard(seq)
            try:
                os.remove(self.dir / name)
            except OSError:
                pass

    def _write_playlist(self, ended=False):
        with self.lock:
            text = playlist_text(self.segments, ended=ended, discontinuities=self.discontinuities)
        tmp = str(self.playlist_path) + '.tmp'
        with open(tmp, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write(text)
        # On Windows the replace is refused while a reader holds the file open
        # (send_file streams it to a player): a few short retries, never a
        # feeder killed by its last write.
        for attempt in range(20):
            try:
                os.replace(tmp, self.playlist_path)
                return
            except PermissionError:
                time.sleep(0.05 * (attempt + 1))
        os.replace(tmp, self.playlist_path)

    # -- viewer ----------------------------------------------------------------

    def note_segment_request(self, name):
        """A player asked for this segment: that is where the viewer is."""
        m = re.match(r'^seg_(\d{6})\.ts$', str(name or ''))
        if not m:
            return
        seq = int(m.group(1))
        with self.lock:
            if seq > self.last_requested_seq:
                self.last_requested_seq = seq

    # -- lifecycle -------------------------------------------------------------

    def start(self):
        for target, name in ((self._produce, 'producer'), (self._feed, 'feeder')):
            t = threading.Thread(target=target, name=f'live-{name}-{self.id}', daemon=True)
            t.start()
            self.threads.append(t)

    def stop(self):
        if self.state in ('stopping', 'stopped'):
            return   # a second Stop must not revive 'stopping' on a channel already gone
        self.state = 'stopping'
        self.stop_event.set()
        with self.lock:
            inflight = list(self.inflight)
            self.inflight.clear()
        # Give back the clips still WAITING in the queue. The one already on
        # the card is left to finish, and its output is discarded when it
        # lands: interrupting it raises the queue's recovery barrier, which
        # blocks every GPU action — a new channel included — until the prompt
        # is proven gone. Measured: a restart right after a stop answered 409.
        from ..job_queue import queue_manager
        with self.app.app_context():
            for job_id in inflight:
                if self._job_status(job_id) == 'pending':
                    try:
                        queue_manager.cancel_job(job_id, self.user_id)
                        continue
                    except Exception:  # noqa: BLE001 — a cancel that fails is a clip that lands late, and is discarded
                        logger.debug('live %s: cancel of %s failed', self.id, job_id, exc_info=True)
                with self.lock:
                    self.abandoned.add(job_id)


# ── Module-level singleton ──────────────────────────────────────────────────

_session = None
_session_lock = threading.Lock()
# The last few channels by id: a clip left on the card at a stop lands after
# its channel was replaced by the next start, and must reach the one that
# abandoned it (to be discarded), not the one now open (which ignores it).
_recent = collections.OrderedDict()
RECENT_SESSIONS = 4


def current():
    return _session


def start(app, user_id, params) -> LiveSession:
    """Open the channel. Refuses while one is running."""
    global _session
    with _session_lock:
        if _session is not None and _session.state in ('starting', 'running', 'stopping'):
            raise LiveError('A channel is already running — stop it first.')
        facts = ffmpeg_facts()
        if not facts.get('path'):
            raise LiveError('ffmpeg is not available — the stream cannot be encoded.')
        _sweep_root()
        session = LiveSession(app, user_id, params)
        session.start()
        _session = session
        _recent[session.id] = session
        while len(_recent) > RECENT_SESSIONS:
            _recent.popitem(last=False)
        return session


def stop() -> dict | None:
    with _session_lock:
        s = _session
    if s is None:
        return None
    s.stop()
    return s.status()


def link_completed_live_clip(job_id, filename, failed=False, reason=None, session_id=None):
    """The queue's completion callback for a live clip (see _dispatch_completion).
    `session_id` is the channel that queued it: the one open now may be its
    successor."""
    s = _recent.get(session_id) if session_id else None
    if s is None:
        s = current()
    if s is None:
        logger.info('live: completion for job %s with no channel open — ignored', job_id)
        return
    s.on_completed(job_id, filename, failed, reason, _render_seconds(job_id))


def _render_seconds(job_id):
    """Claim → settled of the queue row: the same measurement as the Studio's
    clip cards, for the same reason (model loading included)."""
    from ..models import ImageGenerationQueue
    try:
        job = ImageGenerationQueue.query.filter_by(job_id=job_id).first()
    except Exception:  # noqa: BLE001 — a session error must not drop the clip
        return None
    if job is None or job.status not in ('completed', 'failed'):
        return None
    if not job.started_at or not job.completed_at:
        return None
    secs = (job.completed_at - job.started_at).total_seconds()
    return round(secs, 1) if secs >= 0 else None


def _sweep_root():
    """Previous channels' segments: dead weight the moment a new one opens."""
    root = live_root(create=True)
    for child in root.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink()
        except OSError:
            pass
    _sweep_comfy_output()


# What a live clip is called in ComfyUI's output folder: the Studio's prefix,
# this lane's tag and the channel id, then ComfyUI's own counter.
_LIVE_OUTPUT_RE = re.compile(r'^\w*lds_video_test_\w+_live_[0-9a-f]{8}_\d+_\.[A-Za-z0-9]+$')


def _sweep_comfy_output():
    """Live clips a previous channel left in ComfyUI's output folder — a clip
    on the card when the app was closed lands with nobody to discard it."""
    from . import lora_test_studio as lts
    try:
        out_dir = lts._comfy_output_dir()
    except Exception:  # noqa: BLE001 — no folder to sweep is not an error
        return
    if not out_dir or not os.path.isdir(out_dir):
        return
    for name in os.listdir(out_dir):
        if _LIVE_OUTPUT_RE.match(name):
            try:
                os.remove(os.path.join(out_dir, name))
                logger.info('live: swept %s, left by a previous channel', name)
            except OSError:
                pass
