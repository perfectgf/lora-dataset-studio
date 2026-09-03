/* 🔴 The live channel's API shape and the sentences the panel says.
 *
 * Pure, so `node --test` covers it without a DOM: the URLs, the payload the
 * Start button sends, the pace line built from the server's status, and the
 * address a VLC on the LAN opens. The component only wires these to state. */
export const LIVE_BASE = '/api/video-studio/live';
export const liveOptionsUrl = () => `${LIVE_BASE}/options`;
export const liveStartUrl = () => `${LIVE_BASE}/start`;
export const liveStopUrl = () => `${LIVE_BASE}/stop`;
export const liveStatusUrl = () => `${LIVE_BASE}/status`;

/* Playback rates offered. `auto` lets the server pick a tenth under what the
   card sustains, after the prefill; the numbers are what a person would
   choose knowing that H3 authors motion at 24 fps (18 = 75 % speed, 12 =
   half, 6 = a quarter — still a picture that moves). */
export const FPS_CHOICES = ['auto', 24, 18, 16, 12, 10, 8, 6];
export const FPS_MIN = 6;   // the stream's floor — mirrors the backend's FPS_MIN
export const ASPECTS = [
  { id: 'landscape', label: '16:9' }, { id: 'portrait', label: '9:16' }, { id: 'square', label: '1:1' },
];

export const isLiveRunning = (status) =>
  !!status && ['starting', 'running', 'stopping'].includes(status.state);

/** How many scenes a text holds — blocks between `---` lines, blank ones dropped. */
export function sceneCount(text) {
  return String(text || '').split(/^\s*---\s*$/m).filter((b) => b.trim()).length;
}

/** The Start payload. `auto` → 0 (the server's own word for it); an option
    left empty is left OUT, never sent as false. */
export function buildStartPayload(state) {
  const out = {
    scenes: state.scenes || '',
    subject: state.subject || '',
    megapixels: Number(state.megapixels),
    aspect: state.aspect || 'landscape',
    frames: Number(state.frames),
    fps: state.fps === 'auto' || state.fps === '' || state.fps == null ? 0 : Number(state.fps),
    turbo: state.turbo !== false,
  };
  if (state.lora) {
    out.lora = state.lora;
    out.lora_strength = Number(state.loraStrength ?? 1);
  }
  if (state.steps !== '' && state.steps != null) out.steps = Number(state.steps);
  if (state.seed !== '' && state.seed != null) out.seed = Number(state.seed);
  if (state.eros) out.eros = true;
  if (state.sparse) out.sparse = state.sparse;
  return out;
}

/** The absolute address of the playlist — what VLC on another machine opens.
    Same host and port as the app: the guard that protects the app protects
    the stream, and a LAN that needs the access token needs it here too. */
export function streamUrlFor(status, origin) {
  if (!status || !status.playlist || !origin) return null;
  return `${origin}${status.playlist}`;
}

/** The write-up's per-clip line, as one sentence for the rail. */
export function paceLine(status) {
  if (!status || status.state === 'idle') return '';
  const n = status.produced || 0;
  if (status.state === 'stopped') {
    return `Channel stopped — ${n} clip${n === 1 ? '' : 's'} streamed.`;
  }
  if (status.pace === 'measuring' || !status.play_fps) {
    // Measured clips, not encoded segments: nothing is encoded before the rate
    // is decided, so `produced` reads 0 through the whole prefill.
    const measured = status.measured ?? n;
    const need = Math.max(0, 2 - measured);
    return `Measuring the pace — ${measured} clip${measured === 1 ? '' : 's'} measured`
      + (status.play_fps ? '' : `, playback starts after ${need} more.`);
  }
  const fps = Number(status.play_fps);
  const speed = Math.round((fps / 24) * 100);
  const head = `Playing at ${fps} fps (motion at ${speed} % speed): a clip plays for `
    + `${status.play_seconds} s and renders in ${status.render_seconds} s — the card sustains `
    + `${status.sustain_fps} fps.`;
  if (status.pace === 'keeping_up') {
    return `${head} Keeping up, ${status.margin_seconds} s of buffer gained per clip.`;
  }
  const runway = status.runway_clips;
  // A clip pays a fixed cost (model call, decode, encode) whatever its length:
  // more frames per clip buy more seconds of playback per second of render.
  // Shorter clips never help, and at the floor there is no lower rate to pick.
  const advice = fps <= FPS_MIN
    ? 'This is the lowest rate: lengthen the clips or drop the resolution.'
    : 'Pick a lower rate, lengthen the clips or drop the resolution.';
  return `${head} Behind by ${Math.abs(status.margin_seconds)} s per clip — `
    + (runway ? `${runway} clip${runway === 1 ? '' : 's'} of buffer left. ` : 'the player waits between clips. ')
    + advice;
}

/** One sentence when the producer waits for the player, or nothing. */
export function pauseLine(status) {
  if (!status || !status.paused_for_viewer || !isLiveRunning(status)) return '';
  return 'Rendering is paused until the player catches up — nothing is spent on clips nobody watches.';
}
