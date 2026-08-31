/**
 * 🎬 The Video Test Studio — the video lane's answer to "is this LoRA any good".
 *
 * WHY THIS IS NOT A GRID
 * The image studio's whole shape is a matrix: checkpoints × strengths, twelve
 * cells, twelve seconds, and the answer is in the contact sheet. A clip is
 * minutes. The same shape here would be half an hour of waiting before anything
 * could be looked at, so this queues ONE clip per launch and keeps a history —
 * comparison happens in time (two players, same seed, one setting changed)
 * rather than in space.
 *
 * THE SHAPE OF THE SCREEN (redesign, 2026-08-31 — "respecte le thème général")
 * A take sheet. On a wide screen the TAKE sits on the left — which LoRA, which
 * start frame, what moves — and the RENDER rail on the right stays in view
 * while you scroll: the dials and the Generate button, with a one-line readback
 * of exactly what is about to be rendered. Below, full width, the clips. On a
 * phone everything stacks and the same fixed StudioActionBar the image lane
 * uses keeps Generate one thumb away — one vocabulary for both lanes, no chrome
 * invented for this one. The first build was a flat stack of identical cards
 * with the button at the bottom, grey, under twenty rows of LoRA files.
 *
 * WHAT IT SHARES WITH THE IMAGE STUDIO
 * The queue, the missing-asset refusal, the completion callback and the LoRA
 * safety guard are all the same code. The pipeline underneath is the MiniMax H3
 * image-to-video graph this project's own video generation has been running for
 * months; nothing about the engine was reinvented for this panel.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { Clapperboard, Play } from 'lucide-react';
import { apiFetch, del, postJson } from '../../../../api/fetchClient';
import { HelpBadge } from '../../../../help/HelpMode';
import { useToast } from '../../../common/Toast';
import StudioActionBar from '../StudioActionBar';
import VideoClipHistory from './VideoClipHistory';
import VideoLoraPicker from './VideoLoraPicker';
import VideoOptionsPanel from './VideoOptionsPanel';
import VideoSourcePicker from './VideoSourcePicker';
import { shortLoraName } from './videoLoraGroups';
import {
  buildGeneratePayload, clipRateUrl, clipSeconds, clipUrl, clipsUrl, generateUrl,
  isRunning, optionsUrl,
} from './videoStudioApi';

/* Turbo ON by default. Without it the base is undistilled and a first clip is
   tens of minutes — long enough that a new user concludes the studio is broken
   rather than slow. It is a checkbox, and the panel says what it changes. */
const DEFAULT_OPTIONS = {
  turbo: true, eros: false, sparse: '', latentUpscale: false,
  frames: 56, megapixels: 0.3, seed: '',
};

/* The bottom bar's jump targets — the sections of the take sheet, in the
   order you fill them. Same component and same idiom as the image lane. */
const SHORTCUTS = [
  { id: 'vs-lora', emoji: '🧬', label: 'LoRA' },
  { id: 'vs-source', emoji: '🖼', label: 'Start frame' },
  { id: 'vs-motion', emoji: '✍', label: 'Motion' },
  { id: 'vs-render', emoji: '⚙', label: 'Render' },
  { id: 'vs-clips', emoji: '🎞', label: 'Clips' },
];

export default function VideoTestStudio() {
  const toast = useToast();
  const [options, setOptions] = useState(null);
  const [lora, setLora] = useState({ lora: null, runId: null, datasetId: null });
  const [strength, setStrength] = useState(1.3);
  const [mode, setMode] = useState('i2v');
  const [aspect, setAspect] = useState('landscape');
  const [source, setSource] = useState({ image: null, ratio: null, preview: null });
  const [prompt, setPrompt] = useState('');
  const [opts, setOpts] = useState(DEFAULT_OPTIONS);
  const [clips, setClips] = useState([]);
  const [busy, setBusy] = useState(false);
  const pollRef = useRef(null);

  useEffect(() => {
    apiFetch(optionsUrl()).then((d) => {
      setOptions(d);
      if (d?.frame_default) setOpts((o) => ({ ...o, frames: d.frame_default }));
      // Turbo defaults ON, but only where it CAN run: on a ComfyUI without the
      // pack it would send a launch that is refused before anything happens,
      // which is a poor first click. `available === null` (probe unreachable)
      // keeps the default — an unknown is not a no.
      if (d?.options_available?.turbo?.available === false) {
        setOpts((o) => ({ ...o, turbo: false }));
      }
      if (d?.megapixels?.default) {
        setOpts((o) => ({ ...o, megapixels: d.megapixels.default }));
      }
    }).catch(() => setOptions(null));
  }, []);

  const refreshClips = useCallback(async () => {
    try {
      const d = await apiFetch(clipsUrl(24));
      setClips(d.clips || []);
      return d.clips || [];
    } catch {
      return [];
    }
  }, []);
  useEffect(() => { refreshClips(); }, [refreshClips]);

  /* Poll only while something is actually rendering, and stop the moment
     nothing is: a clip takes minutes, and a timer that keeps firing on an idle
     panel is a request every three seconds for as long as the tab is open. */
  useEffect(() => {
    const running = clips.some(isRunning);
    if (!running) {
      if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
      return undefined;
    }
    if (pollRef.current) return undefined;
    pollRef.current = setInterval(refreshClips, 3000);
    return () => {
      if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
    };
  }, [clips, refreshClips]);

  const generate = async () => {
    setBusy(true);
    try {
      const body = buildGeneratePayload({
        mode, prompt, image: source.image, ratio: source.ratio, aspect,
        lora: lora.lora, loraStrength: strength, runId: lora.runId,
        datasetId: lora.datasetId, ...opts,
      });
      const r = await postJson(generateUrl(), body);
      toast.success(`Queued — seed ${r.seed}, ${r.frames} frames.`);
      refreshClips();
    } catch (e) {
      toast.error(e?.message || 'The clip could not be queued.');
    } finally {
      setBusy(false);
    }
  };

  const rate = async (clip, rating) => {
    try {
      const r = await postJson(clipRateUrl(clip.id), { rating });
      setClips((cs) => cs.map((c) => (c.id === clip.id ? { ...c, rating: r.rating } : c)));
    } catch (e) {
      toast.error(e?.message || 'Could not save that.');
    }
  };

  const remove = async (clip) => {
    try {
      await del(clipUrl(clip.id));
      setClips((cs) => cs.filter((c) => c.id !== clip.id));
    } catch (e) {
      toast.error(e?.message || 'Could not delete that clip.');
    }
  };

  /* Reuse loads a past clip's settings back into the panel — including its
     SEED, which is the whole point: changing one dial on the same seed is the
     only comparison that says anything about the dial. */
  const reuse = (clip) => {
    setPrompt(clip.prompt || '');
    setMode(clip.mode === 't2v' ? 't2v' : 'i2v');
    setOpts({
      turbo: !!clip.turbo, eros: !!clip.eros, sparse: clip.sparse || '',
      latentUpscale: !!clip.latent_upscale, frames: clip.frames || opts.frames,
      megapixels: clip.megapixels || opts.megapixels,
      seed: clip.seed ?? '',
    });
    if (clip.lora) {
      setLora({ lora: clip.lora, runId: clip.run_id, datasetId: clip.dataset_id });
      setStrength(clip.lora_strength ?? 1);
    }
    toast.info?.('Settings loaded — change one thing and generate again.');
  };

  const needsImage = mode === 'i2v' && !source.image;
  const blocked = busy || needsImage || !prompt.trim();
  const reason = needsImage
    ? 'Pick a start frame, or switch to text-only.'
    : (!prompt.trim() ? 'Describe the motion first.' : null);

  /* The readback: what is about to be rendered, in one line, next to the
     button — the moment before a multi-minute job is the moment to catch
     "wrong LoRA" or "still on 10Eros". */
  const fps = options?.fps || 24;
  const seconds = clipSeconds(opts.frames, fps);
  const readback = [
    lora.lora ? `${shortLoraName(lora.lora)} @ ${Number(strength).toFixed(2)}` : 'no LoRA',
    mode === 't2v' ? 'text only' : 'from an image',
    seconds ? `${seconds}s` : `${opts.frames} frames`,
    `${Number(opts.megapixels).toFixed(2)} MP`,
    opts.turbo ? 'turbo' : null,
    opts.eros ? '10Eros' : null,
    opts.sparse ? `sparse ${opts.sparse}` : null,
    opts.latentUpscale ? 'upscale ×2' : null,
  ].filter(Boolean).join(' · ');

  const generateButton = (
    <button type="button" onClick={generate} disabled={blocked}
      className="flex w-full items-center justify-center gap-2 rounded-lg bg-gradient-primary px-4 py-2 text-sm font-semibold text-gray-950 disabled:opacity-40 min-h-10">
      <Play aria-hidden="true" className="h-4 w-4" />
      {busy ? 'Queueing…' : 'Generate clip'}
    </button>
  );

  return (
    <div className="flex flex-col gap-3">
      <header data-probe-chrome="video-studio-header"
        className="flex flex-wrap items-center gap-2">
        <h2 className="flex items-center gap-2 font-bold text-content">
          <Clapperboard aria-hidden="true" className="h-4 w-4" />
          Video Test Studio
          <HelpBadge topic="page-video-studio" />
        </h2>
        <span className="rounded-lg border border-amber-400/40 bg-amber-400/10 px-2 py-0.5 text-[0.6875rem] font-semibold text-amber-200">
          MiniMax H3 · beta
        </span>
        <span className="hidden text-xs text-content-subtle sm:inline">
          One clip per launch — compare in time, same seed, one dial changed.
        </span>
      </header>

      {/* The one banner that has to come BEFORE everything else: on a machine
          without the weights, every control below is a promise the lane cannot
          keep. It names the missing files by what they do and points at the one
          screen that installs them. */}
      {options && options.ready === false && (
        <div className="rounded-xl border border-amber-400/40 bg-amber-400/10 p-3 text-sm">
          <p className="font-semibold text-amber-200">This lane is not installed yet</p>
          <p className="mt-1 text-content-muted">
            {(options.missing_weights || []).filter((m) => m.required).length} required
            file(s) are missing — about 39.5 GB in total. Install them from the
            Setup screen, under 🎬 Video Test Studio.
          </p>
          <ul className="mt-1 list-disc pl-5 text-[0.6875rem] text-content-subtle">
            {(options.missing_weights || []).filter((m) => m.required).map((m) => (
              <li key={m.filename}>{m.what} — <code className="break-all">{m.filename}</code></li>
            ))}
          </ul>
        </div>
      )}

      <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_22.5rem] lg:items-start">
        {/* THE TAKE — what is being tested, top to bottom in the order you
            decide it. */}
        <div className="flex min-w-0 flex-col gap-3">
          <div id="vs-lora" className="scroll-mt-16">
            <VideoLoraPicker value={lora.lora} onChange={setLora}
              strength={strength} onStrength={setStrength} />
          </div>

          <div id="vs-source" className="scroll-mt-16">
            <VideoSourcePicker mode={mode} onMode={setMode} image={source.image}
              preview={source.preview} aspect={aspect} onAspect={setAspect}
              onPicked={setSource} />
          </div>

          <label id="vs-motion" className="flex flex-col gap-1.5 rounded-xl border border-border bg-surface p-3 scroll-mt-16">
            <span className="text-sm font-semibold text-content">Motion</span>
            <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={3}
              placeholder="What happens in the shot — she turns her head and smiles, the camera pushes in slowly…"
              className="w-full resize-y rounded-lg border border-border bg-app px-2.5 py-2 text-sm text-content" />
            <span className="text-[0.6875rem] text-content-subtle">
              Describe the movement, not the picture: the start frame already says
              what the scene looks like.
            </span>
          </label>
        </div>

        {/* THE RENDER RAIL — sticky on a wide screen so the dials and the
            button never leave the eye while the take scrolls. */}
        <aside id="vs-render" className="flex min-w-0 flex-col gap-3 scroll-mt-16 lg:sticky lg:top-3">
          <VideoOptionsPanel options={options} value={opts} onChange={setOpts} />
          <div className="flex flex-col gap-2 rounded-xl border border-border bg-surface p-3">
            <p className="break-words font-mono text-[0.6875rem] leading-snug text-content-muted">
              {readback}
            </p>
            {generateButton}
            {reason && (
              <p className="text-[0.6875rem] text-content-subtle">{reason}</p>
            )}
          </div>
        </aside>
      </div>

      <section id="vs-clips" className="flex flex-col gap-2 scroll-mt-16">
        <h2 className="font-mono text-[0.625rem] uppercase tracking-[0.18em] text-content-subtle">
          Clips — newest first
        </h2>
        <VideoClipHistory clips={clips} onRate={rate} onDelete={remove} onReuse={reuse} />
      </section>

      <StudioActionBar shortcuts={SHORTCUTS} canRun={!blocked} running={busy}
        onRun={generate} runLabel="▶ Generate clip" note={reason} />
    </div>
  );
}
