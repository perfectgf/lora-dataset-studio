/**
 * 🔴 Live — a channel that never stops, from the LoRA you are testing.
 *
 * The Video lane renders ONE clip and compares. This lane is the other shape
 * of the same engine: scenes are drawn from a list, clips are rendered back to
 * back, and every finished one is appended to a stream a player reads like a
 * TV channel — here in the browser, or in VLC on any machine of the LAN. The
 * shape comes from FastH3 Live (jacokon, Apache-2.0): generation slower than
 * playback on one card, so the stream is retimed on the way out and the rail
 * says what the card sustains. Experimental: the pipeline is the Studio's own,
 * the numbers are measured on this machine, and a 24 GB card will not reach
 * real time — it will reach a channel.
 *
 * SAME BONES AS THE VIDEO LANE: the take on the left (LoRA, scenes, subject),
 * the rail on the right (dials, the Start button, the pace line), the player
 * full width under both. Nothing invented for this screen.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { Copy, Radio, Square } from 'lucide-react';
import { apiFetch, postJson } from '../../../../api/fetchClient';
import { useToast } from '../../../common/Toast';
import { HelpBadge } from '../../../../help/HelpMode';
import VideoLoraPicker from '../video/VideoLoraPicker';
import { shortLoraName } from '../video/videoLoraGroups';
import { clipSeconds, optionsUrl, studioFrameChoices } from '../video/videoStudioApi';
import {
  ASPECTS, FPS_CHOICES, buildStartPayload, isLiveRunning, liveOptionsUrl, liveStartUrl, pauseLine,
  liveStatusUrl, liveStopUrl, paceLine, sceneCount, streamUrlFor,
} from './liveStudioApi';

const POLL_MS = 3000;
const DEFAULT_FRAMES = 124;   // ~5 s at 24 fps: short enough to loop, long enough to watch

export default function LiveStudio() {
  const toast = useToast();
  const [options, setOptions] = useState(null);       // the Video lane's own options
  const [liveOptions, setLiveOptions] = useState(null);
  const [lora, setLora] = useState({ lora: null, runId: null, datasetId: null });
  const [strength, setStrength] = useState(1.0);
  const [scenes, setScenes] = useState('');
  const [subject, setSubject] = useState('');
  const [dials, setDials] = useState({ megapixels: 0.3, aspect: 'landscape', frames: DEFAULT_FRAMES,
    fps: 'auto', steps: 6, turbo: true });   // 6: the turbo LoRA's own step count (TURBO_STEPS), as the Video lane
  const [status, setStatus] = useState(null);
  const [pollFailed, setPollFailed] = useState(false);   // the status poll cannot reach the app
  const [playerError, setPlayerError] = useState(null);  // what the browser player could not do
  const [busy, setBusy] = useState(false);
  const videoRef = useRef(null);
  const hlsRef = useRef(null);
  const set = (patch) => setDials((d) => ({ ...d, ...patch }));

  useEffect(() => {
    apiFetch(optionsUrl()).then((d) => {
      setOptions(d);
      if (d?.megapixels?.default) set({ megapixels: d.megapixels.default });
    }).catch(() => setOptions(null));
    apiFetch(liveOptionsUrl()).then((d) => {
      setLiveOptions(d);
      setScenes((s) => s || d?.default_scenes || '');
    }).catch(() => setLiveOptions(null));
  }, []);

  const refresh = useCallback(async () => {
    try {
      const d = await apiFetch(liveStatusUrl(), { background: true });   // a poll: no toast per miss
      setStatus(d);
      setPollFailed(false);
      return d;
    } catch {
      setPollFailed(true);   // keep the last status, but say it is stale
      return null;
    }
  }, []);
  useEffect(() => { refresh(); }, [refresh]);

  /* Poll only while the channel lives: a stopped or idle channel is a request
     every three seconds for nothing. */
  useEffect(() => {
    if (!isLiveRunning(status)) return undefined;
    const t = setInterval(refresh, POLL_MS);
    return () => clearInterval(t);
  }, [status, refresh]);

  /* The player. hls.js is loaded only when there is something to play — it is
     a quarter megabyte the Images lane never needs — and it comes FIRST
     wherever it runs: Chrome answers "maybe" to the HLS MIME and would get a
     native player that cannot play it; the native path is Safari's. The
     player outlives the channel: ENDLIST lets it drain what it has, and it is
     torn down when the playlist changes (a new channel) or the panel goes. */
  const playlist = status?.playlist && status.segments > 0 && status.state !== 'idle' ? status.playlist : null;
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !playlist) return undefined;
    let cancelled = false;
    setPlayerError(null);
    (async () => {
      let Hls = null;
      try {
        ({ default: Hls } = await import('hls.js'));
      } catch {
        if (!cancelled) setPlayerError('The player could not be loaded (hls.js) — open the address in VLC.');
        return;
      }
      if (cancelled) return;
      if (Hls.isSupported()) {
        const hls = new Hls({ liveSyncDurationCount: 2, liveDurationInfinity: true });
        hlsRef.current = hls;
        hls.on(Hls.Events.ERROR, (_event, data) => {
          if (data?.fatal) setPlayerError(`The player stopped: ${data.details || data.type}. Reload the tab or open the address in VLC.`);
        });
        hls.loadSource(playlist);
        hls.attachMedia(video);
      } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
        video.src = playlist;
      } else {
        setPlayerError('This browser cannot play the stream — open the address in VLC.');
      }
    })();
    return () => {
      cancelled = true;
      if (hlsRef.current) { hlsRef.current.destroy(); hlsRef.current = null; }
      video.removeAttribute('src');   // the native path too, or the two players diverge
      video.load();
    };
  }, [playlist]);

  const start = async () => {
    setBusy(true);
    try {
      const body = buildStartPayload({ scenes, subject, lora: lora.lora, loraStrength: strength, ...dials });
      const r = await postJson(liveStartUrl(), body);
      setStatus(r);
      toast.success('Channel open — the first clips are rendering.');
    } catch (e) {
      toast.error(e?.message || 'The channel could not start.');
    } finally {
      setBusy(false);
    }
  };
  const stop = async () => {
    setBusy(true);
    try {
      const r = await postJson(liveStopUrl(), {});
      setStatus(r);
    } catch (e) {
      toast.error(e?.message || 'The channel could not be stopped.');
    } finally {
      setBusy(false);
    }
  };

  const running = isLiveRunning(status);
  const origin = typeof window !== 'undefined' ? window.location.origin : '';
  const vlcUrl = streamUrlFor(status, origin);
  // The playlist exists once the first segment does: an address shown before
  // that is a 404 for the whole prefill.
  const streamReady = !!vlcUrl && (status?.segments || 0) > 0;
  const copyUrl = async () => {
    try {
      await navigator.clipboard.writeText(vlcUrl);
      toast.success('Stream address copied — open it in VLC (Media › Open Network Stream).');
    } catch {
      toast.info?.(vlcUrl);
    }
  };
  const frames = studioFrameChoices(options);
  const fps = options?.fps || 24;
  const nScenes = sceneCount(scenes);
  const canStart = !busy && !running && nScenes > 0 && options?.ready !== false
    && liveOptions?.ffmpeg !== false;
  const seconds = clipSeconds(dials.frames, fps);
  // While the channel runs, the LoRA line reads the server's own parameters:
  // the picker is frozen, and what renders is what was sent, not what is shown.
  const onAir = running && status?.params ? status.params : null;
  const loraLine = onAir
    ? (onAir.lora ? `${shortLoraName(onAir.lora)} @ ${Number(onAir.lora_strength ?? 1).toFixed(2)}` : 'no LoRA')
    : (lora.lora ? `${shortLoraName(lora.lora)} @ ${Number(strength).toFixed(2)}` : 'no LoRA');
  const readback = [
    loraLine,
    `${nScenes} scene${nScenes === 1 ? '' : 's'}`,
    seconds ? `${seconds}s clips` : `${dials.frames} frames`,
    `${Number(dials.megapixels).toFixed(2)} MP`,
    dials.fps === 'auto' ? 'auto rate' : `${dials.fps} fps`,
    dials.turbo ? `turbo ${dials.steps} steps` : `${dials.steps} steps`,
  ].join(' · ');

  return (
    <div className="flex flex-col gap-3" data-testid="live-studio">
      <header data-probe-chrome="live-studio-header"
        className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="flex items-center gap-2 text-lg font-semibold">
          <Radio aria-hidden="true" className="h-5 w-5 text-primary" /> Live channel
          <span className="rounded-full border border-amber-400/40 bg-amber-400/10 px-2 py-px text-[0.625rem] font-semibold uppercase tracking-wide text-amber-200">beta</span>
          <HelpBadge topic="page-video-live" />
        </h2>
        <span className="hidden text-xs text-content-subtle sm:inline">
          Clips rendered back to back from your scenes, streamed as they land — here, or in VLC.
        </span>
      </header>

      {options && options.ready === false && (
        <div className="rounded-xl border border-amber-400/40 bg-amber-400/10 p-3 text-sm">
          <p className="font-semibold text-amber-200">The video lane is not installed yet</p>
          <p className="mt-1 text-content-muted">
            The channel renders with the Video Test Studio&apos;s engine: install its weights from
            the Setup screen, under 🎬 Video Test Studio, then come back.
          </p>
        </div>
      )}
      {liveOptions && liveOptions.ffmpeg === false && (
        <div className="rounded-xl border border-red-500/40 bg-red-500/10 p-3 text-sm">
          <p className="font-semibold">ffmpeg is not available</p>
          <p className="mt-1 text-content-muted">The stream is encoded by ffmpeg; without it the channel cannot open.</p>
        </div>
      )}

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-[minmax(0,1fr)_22.5rem] lg:items-start">
        <div className="flex min-w-0 flex-col gap-3" data-probe-panel="live-take">
          {/* Frozen while the channel runs: a change here would show a LoRA the
              clips are not rendered with. */}
          <div className={running ? 'pointer-events-none opacity-60' : undefined} aria-disabled={running || undefined}>
            <VideoLoraPicker value={lora.lora} onChange={setLora}
              strength={strength} onStrength={setStrength} />
          </div>

          <section className="rounded-xl border border-border bg-surface p-3">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <h3 className="text-sm font-semibold">Scenes</h3>
              <span className="text-[0.6875rem] text-content-subtle">
                one per block, separated by a line with <code>---</code> · <code>{'{NAME}'}</code> becomes the subject
              </span>
            </div>
            <textarea value={scenes} onChange={(e) => setScenes(e.target.value)}
              rows={10} spellCheck={false} disabled={running}
              aria-label="Scenes"
              className="mt-2 w-full resize-y rounded-lg border border-border bg-surface-raised p-2 font-mono text-xs leading-relaxed text-content" />
            <label className="mt-2 flex flex-wrap items-center gap-2 text-xs">
              <span className="text-content-muted">Subject</span>
              <input type="text" value={subject} onChange={(e) => setSubject(e.target.value)}
                placeholder="the trigger word of your LoRA, or who the scenes are about"
                disabled={running} aria-label="Subject"
                className="min-w-0 flex-1 rounded-lg border border-border bg-surface-raised px-2 py-1 text-xs min-h-10 lg:min-h-0" />
            </label>
            <p className="mt-1 text-[0.6875rem] text-content-subtle">
              Scenes are drawn in a shuffled order and never repeat until every one has played.
              Write them in H3&apos;s own grammar — what is shown, then the soundscape.
            </p>
          </section>
        </div>

        <aside className="flex flex-col gap-3 lg:sticky lg:top-16" data-probe-panel="live-rail">
          <section className="rounded-xl border border-border bg-surface p-3">
            <h3 className="text-sm font-semibold">Channel</h3>
            <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
              <label className="flex flex-col gap-1">
                <span className="text-content-muted">Clip length</span>
                <select value={dials.frames} onChange={(e) => set({ frames: Number(e.target.value) })}
                  disabled={running} aria-label="Clip length"
                  className="rounded-lg border border-border bg-surface-raised px-2 py-1 min-h-10 lg:min-h-0">
                  {frames.map((f) => (
                    <option key={f} value={f}>{clipSeconds(f, fps)}s · {f} frames</option>
                  ))}
                </select>
              </label>
              <label className="flex flex-col gap-1">
                <span className="text-content-muted">Playback rate</span>
                <select value={dials.fps} onChange={(e) => set({ fps: e.target.value === 'auto' ? 'auto' : Number(e.target.value) })}
                  disabled={running} aria-label="Playback rate"
                  className="rounded-lg border border-border bg-surface-raised px-2 py-1 min-h-10 lg:min-h-0">
                  {FPS_CHOICES.map((f) => (
                    <option key={f} value={f}>{f === 'auto' ? 'auto (what the card sustains)' : `${f} fps · ${Math.round((f / 24) * 100)} % speed`}</option>
                  ))}
                </select>
              </label>
              <label className="flex flex-col gap-1">
                <span className="text-content-muted">Aspect</span>
                <select value={dials.aspect} onChange={(e) => set({ aspect: e.target.value })}
                  disabled={running} aria-label="Aspect"
                  className="rounded-lg border border-border bg-surface-raised px-2 py-1 min-h-10 lg:min-h-0">
                  {ASPECTS.map((a) => <option key={a.id} value={a.id}>{a.label}</option>)}
                </select>
              </label>
              <label className="flex flex-col gap-1">
                <span className="text-content-muted">Resolution · {Number(dials.megapixels).toFixed(2)} MP</span>
                <input type="range" min={options?.megapixels?.min ?? 0.1} max={options?.megapixels?.max ?? 2}
                  step="0.05" value={dials.megapixels} disabled={running} aria-label="Resolution"
                  onChange={(e) => set({ megapixels: Number(e.target.value) })} className="min-h-10 lg:min-h-0" />
              </label>
              <label className="flex flex-col gap-1">
                <span className="text-content-muted">Steps</span>
                <input type="number" min={4} max={40} value={dials.steps} disabled={running} aria-label="Steps"
                  onChange={(e) => set({ steps: Number(e.target.value) || 4 })}
                  className="rounded-lg border border-border bg-surface-raised px-2 py-1 min-h-10 lg:min-h-0" />
              </label>
              <label className="flex items-center gap-2 self-end min-h-10 lg:min-h-0">
                <input type="checkbox" checked={dials.turbo} disabled={running}
                  onChange={(e) => set({ turbo: e.target.checked })} />
                <span className="text-content-muted">⚡ turbo ({options?.turbo_steps || 6}-step LoRA)</span>
              </label>
            </div>
            <p className="mt-2 text-[0.6875rem] text-content-subtle">
              H3 authors motion at 24 fps. A rate under that plays the same frames slower, with the
              sound stretched to match — the only way one card keeps a channel fed.
            </p>
            <p className="mt-2 truncate text-[0.6875rem] text-content-subtle" title={readback}>{readback}</p>
            <div className="mt-3 flex gap-2">
              {running ? (
                <button type="button" onClick={stop} disabled={busy}
                  className="flex flex-1 items-center justify-center gap-1.5 rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm font-semibold min-h-10">
                  <Square aria-hidden="true" className="h-4 w-4" /> Stop the channel
                </button>
              ) : (
                <button type="button" onClick={start} disabled={!canStart}
                  className="flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-primary px-3 py-2 text-sm font-semibold text-white disabled:opacity-50 min-h-10">
                  <Radio aria-hidden="true" className="h-4 w-4" /> Start the channel
                </button>
              )}
            </div>
          </section>

          {status && status.state !== 'idle' && (
            <section className="rounded-xl border border-border bg-surface p-3 text-xs" data-testid="live-status">
              <p className="text-content">{paceLine(status)}</p>
              <p className="mt-1 text-content-subtle">
                {status.produced || 0} rendered · {status.inflight || 0} in the queue · {status.buffered_clips || 0} buffered
                {status.failed ? ` · ${status.failed} failed` : ''}
                {status.last_scene_index != null ? ` · scene ${status.last_scene_index + 1}/${status.scene_count}` : ''}
              </p>
              {pauseLine(status) && <p className="mt-1 text-content-subtle">{pauseLine(status)}</p>}
              {pollFailed && <p className="mt-1 text-amber-200">The channel&apos;s status is unreachable — is the app still running?</p>}
              {status.error && <p className="mt-1 text-red-300">{status.error}</p>}
              {streamReady && (
                <div className="mt-2 flex items-center gap-2">
                  <code className="min-w-0 flex-1 truncate rounded bg-surface-raised px-2 py-1 text-[0.6875rem]" title={vlcUrl}>{vlcUrl}</code>
                  <button type="button" onClick={copyUrl} title="Copy the stream address for VLC"
                    className="rounded-lg border border-border p-1.5 text-content-muted hover:text-content min-h-10 lg:min-h-0">
                    <Copy aria-hidden="true" className="h-3.5 w-3.5" />
                  </button>
                </div>
              )}
              {streamReady && liveOptions?.token_required && (
                <p className="mt-1 text-content-subtle">
                  This app asks other machines for its access token: in VLC, add <code>?token=…</code> to the
                  address (the token is under Settings › Server).
                </p>
              )}
            </section>
          )}
        </aside>
      </div>

      <section className="rounded-xl border border-border bg-black/40 p-2" data-probe-panel="live-player">
        {playerError && <p className="mb-1 px-1 text-xs text-red-300">{playerError}</p>}
        <video ref={videoRef} controls playsInline className="mx-auto max-h-[70vh] w-full rounded-lg bg-black"
          data-testid="live-player" />
        {!playlist && (
          <p className="p-2 text-center text-xs text-content-subtle">
            {running ? 'Prefilling — the player opens when the first clips have landed.'
              : 'The channel plays here once it is started.'}
          </p>
        )}
      </section>
    </div>
  );
}
