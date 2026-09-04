/**
 * 🎬 The Video Test Studio — the video lane's answer to "is this LoRA any good".
 *
 * WHY THIS IS NOT A GRID
 * The image studio's whole shape is a matrix: checkpoints × strengths, twelve
 * cells, twelve seconds, and the answer is in the contact sheet. A clip is
 * minutes. The same shape here would be half an hour of waiting before anything
 * could be looked at, so this queues one clip per START FRAME — a launch is
 * one clip, or one per picture in the strip, all on one seed — and keeps a
 * history: comparison happens in time (two players, same seed, one setting
 * changed) rather than in space.
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
import useOllamaFence from '../../../../hooks/useOllamaFence';
import { SUPERSEDED_ANSWER_NOTICE, keepAnswer } from '../../../../utils/ollamaFence';
import OllamaFenceNotice from '../../../common/OllamaFenceNotice';
import { useToast } from '../../../common/Toast';
import StudioActionBar from '../StudioActionBar';
import VideoClipHistory from './VideoClipHistory';
import VideoLoraPicker from './VideoLoraPicker';
import VideoOptionsPanel from './VideoOptionsPanel';
import VideoQuickPrompts from './VideoQuickPrompts';
import { appendQuickPrompt } from './videoPromptPresets';
import MotionModelDialog from './MotionModelDialog';
import SmoothDialog from './SmoothDialog';
import VideoSourcePicker from './VideoSourcePicker';
import NeuralRenderDialog from '../../../videobank/NeuralRenderDialog';
import SideBySideVideo from '../../../videobank/SideBySideVideo';
import { shortLoraName } from './videoLoraGroups';
import {
  addFrames, failureNotice, generateLabel, perImagePrompts, queueClips, queuedNotice, releasePreview,
  removeFrame,
} from './videoStartFrames';
import {
  accelLabel, clipAccel, clipLastFramePngUrl, clipLastFrameUrl, clipRateUrl, clipSeconds, clipUrl, clipsUrl,
  generateUrl, mergeClipPages,
  pickAvailableAccel,
  isRunning, launchAdviceLines, optionsUrl, clipVfiUrl, clipNeuralRenderUrl, clipVideoUrl,
  clipComparisonUrl,
  motionEnhanceUrl, motionSuggestUrl, motionWriteBatchUrl,
} from './videoStudioApi';

/* No start frame yet — what the ✨ helpers and the readback see before a pick. */
const EMPTY_SOURCE = { image: null, ratio: null, preview: null };

/* An acceleration ON by default — larryvrh's, the arena's first row. Without
   one the base is undistilled and a first clip is tens of minutes — long
   enough that a new user concludes the studio is broken rather than slow. The
   panel says what each choice changes. */
const DEFAULT_OPTIONS = {
  accel: 'turbo', eros: false, sparse: '', latentUpscale: false,
  // '' = auto: the server's own count for the mode in force (turbo 6, dense
  // 20). Kept empty rather than pre-filled so a run reads "auto" until someone
  // decides otherwise — a number in the box would claim a choice nobody made.
  steps: '',
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
  // The start frames, in pick order: the strip the picker draws and the list
  // Generate walks — one clip each, on one seed (one frame was the whole state
  // until 2026-09-02). `source` is the FIRST of them: what ✨ Auto and ✨ Enrich
  // read, and what a change of resets the poller on.
  const [sources, setSources] = useState([]);
  const source = sources[0] || EMPTY_SOURCE;
  const addSources = useCallback((list) => setSources((prev) => addFrames(prev, list).frames), []);
  const removeSource = useCallback((key) => setSources((prev) => removeFrame(prev, key)), []);
  const clearSources = useCallback(() => setSources([]), []);
  // How far a batch is between the click and the last reply, for the button.
  const [progress, setProgress] = useState({ done: 0, total: 0, phase: 'queueing' });
  /* The batch's prompt: ONE for every picture (the default, the comparison
     that says something about the LoRA), or one WRITTEN per picture by ✨ —
     the frame read by the vision model, the typed motion enriched with it or
     a proposal from the picture alone. Written before anything is queued:
     the writer's window shuts once a clip sits in the queue. */
  const [promptMode, setPromptMode] = useState('same');
  const [prompt, setPrompt] = useState('');
  const [opts, setOpts] = useState(DEFAULT_OPTIONS);
  const [clips, setClips] = useState([]);
  const [busy, setBusy] = useState(false);
  const pollRef = useRef(null);

  useEffect(() => {
    apiFetch(optionsUrl()).then((d) => {
      setOptions(d);
      if (d?.frame_default) setOpts((o) => ({ ...o, frames: d.frame_default }));
      // The acceleration defaults to larryvrh's, but only where it CAN run:
      // a launch refused before anything happens is a poor first click. The
      // server says what this machine holds; the pick moves to the first
      // available choice, or to the dense base. `available === null` (probe
      // unreachable) keeps the pick — an unknown is not a no.
      if (Array.isArray(d?.accelerations)) {
        setOpts((o) => ({ ...o, accel: pickAvailableAccel(o.accel, d.accelerations) }));
      }
      if (d?.megapixels?.default) {
        setOpts((o) => ({ ...o, megapixels: d.megapixels.default }));
      }
    }).catch(() => setOptions(null));
  }, []);

  // Whether a page older than what is loaded exists (the server says so).
  const [hasMore, setHasMore] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  /* The newest page REPLACES what it covers and KEEPS what it does not (see
     mergeClipPages): the poll re-reads the first page every three seconds
     while a clip renders, and a poll that replaced the whole list would
     throw away every older page the user had asked for with Load more. The
     boundary is the server's `oldest_id` — the page PROPER, never a source
     that rode along with its render. `oldestLoadedRef` is how far back the
     list reaches, the same boundary lowered by every older page loaded. */
  const oldestLoadedRef = useRef(0);
  const mergeClips = (fresh, keepOlderThan) => setClips((prev) => mergeClipPages(prev, fresh, keepOlderThan));
  const refreshClips = useCallback(async () => {
    try {
      const d = await apiFetch(clipsUrl(24));
      const fresh = d.clips || [];
      const boundary = Number(d.oldest_id) || (fresh.length ? Math.min(...fresh.map((c) => c.id)) : 0);
      mergeClips(fresh, boundary);
      if (!oldestLoadedRef.current || boundary < oldestLoadedRef.current) oldestLoadedRef.current = boundary;
      setHasMore(!!d.has_more);
      return fresh;
    } catch {
      return [];
    }
  }, []);
  const loadMore = useCallback(async () => {
    setLoadingMore(true);
    try {
      const oldest = oldestLoadedRef.current || (clips.length ? Math.min(...clips.map((c) => c.id)) : 0);
      const d = await apiFetch(clipsUrl(24, oldest));
      mergeClips(d.clips || [], 0);
      if (Number(d.oldest_id)) oldestLoadedRef.current = Number(d.oldest_id);
      setHasMore(!!d.has_more);
    } catch {
      toast.error('Could not load older clips.');
    } finally {
      setLoadingMore(false);
    }
  }, [clips, toast]);
  // ↑ Scroll a render's source into view. The server lists it whatever its
  // age, so the card is there; the scroll just finds it.
  const jumpTo = (id) => {
    const el = document.getElementById(`video-clip-${id}`);
    if (el) { el.scrollIntoView({ behavior: 'smooth', block: 'center' }); el.focus?.(); }
    else toast.info?.(`Clip #${id} is no longer in the history.`);
  };
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

  /* One POST per start frame, in order, on one seed and one prompt (the
     server's from the first reply — see queueClips) — text-only is one
     launch without a picture. The walk stops at the first refusal and says how
     far it got; what queued is queued, and the list picks it up. */
  // What a refused write says. A "GPU busy" refusal carries WHY in `detail`
  // (a clip is rendering, training runs) — the same join the dataset passes
  // use, because "GPU busy" alone does not say what to wait for.
  const said = (e, fallback) =>
    [e?.message, e?.body?.detail].filter(Boolean).join(' — ') || fallback;
  /* ✨ One prompt for one picture, the way the two buttons ask: the typed
     motion enriched with the frame, or a proposal from the frame alone.

     ⚠️ Asked for EVERY picture in ONE request, not once per picture. Entering
     the vision window makes ComfyUI let go of its models, so the next clip
     reloads the video model — tens of gigabytes for H3. Twelve single-frame
     calls would pay that twelve times over; `/motion/write-batch` holds one
     window for the whole strip and pays it once. `perImagePrompts` keeps its
     loop and its fallbacks: what changes is WHERE the writing happens. */
  const writePromptsFor = async (frames, typed) => {
    const reply = await postJson(motionWriteBatchUrl(), {
      images: frames.map((f) => f.image),
      prompt: (typed && typed.trim()) ? typed : '',
      model: motionModel, seconds,
    });
    const byIndex = new Map();
    const byImage = new Map();
    for (const r of (reply?.results || [])) {
      if (typeof r?.index === 'number') byIndex.set(r.index, r);
      if (r?.image) byImage.set(r.image, r);
    }
    // The shape `perImagePrompts` expects: resolve to the prompt, or throw the
    // frame's own reason so its fallback and its naming still work.
    return (frame, index) => {
      const r = byIndex.get(index) || byImage.get(frame?.image) || null;
      const written = typeof r?.prompt === 'string' ? r.prompt.trim() : '';
      if (written) return written;
      throw new Error(r?.error || 'the writer had nothing for this picture');
    };
  };

  const generate = async () => {
    setBusy(true);
    let launches = mode === 't2v' ? [null] : sources;
    setProgress({ done: 0, total: launches.length, phase: 'queueing' });
    try {
      const perPicture = mode === 'i2v' && promptMode === 'per-image' && launches.length > 1;
      if (perPicture) {
        setProgress({ done: 0, total: launches.length, phase: 'writing' });
        // ONE request writes for every picture, then the loop below only reads
        // the answers back — no second round trip, no second window.
        const resolve = await writePromptsFor(launches, prompt);
        const written = await perImagePrompts(launches, prompt,
          (frame, typed, index) => resolve(frame, index),
          (done, total) => setProgress({ done, total, phase: 'writing' }));
        launches = written.frames;
        // The pictures the writer could not answer for, BY NAME, and why (a
        // "GPU busy" refusal says what to wait for). All of them: nothing is
        // queued — N renders of a prompt nobody wrote is not a batch.
        const named = written.fallen.map((f) => `picture ${f.index + 1}`).join(', ');
        const why = said(written.error, 'the writer could not answer for them');
        if (written.fallen.length === launches.length) {
          toast.error(`The writer answered for none of the ${launches.length} pictures — ${why}. Nothing was queued.`);
          return;
        }
        if (!prompt.trim()) {
          // No typed motion to fall back on: those pictures sit this batch out.
          launches = launches.filter((f) => f.prompt);
          if (written.fallen.length) toast.warning(`${named} skipped — ${why}.`);
        } else if (written.fallen.length) {
          toast.warning(`${named} launch with the prompt as typed — ${why}.`);
        }
        setProgress({ done: 0, total: launches.length, phase: 'queueing' });
      }
      const outcome = await queueClips(launches, { enhance: enhanceOn && !perPicture,
        mode, prompt, aspect,
        lora: lora.lora, loraStrength: strength, runId: lora.runId,
        datasetId: lora.datasetId, ...opts,
      }, (body) => postJson(generateUrl(), body), (done, total) => setProgress({ done, total, phase: 'queueing' }));
      if (outcome.failed) toast.error(failureNotice(outcome));
      else toast.success(queuedNotice(outcome));
      // The launch went through with the prompt as typed: the writer could
      // not run (fence, server away). Said, or the checkbox looks ignored.
      if (outcome.enrichSkipped) toast.warning(`Queued without enrichment — ${outcome.enrichSkipped}`);
      if (outcome.queued.length) refreshClips();
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
  // ↗ Smoothing. A queued job like any other — the clip list already polls, so
  // the new card simply appears and renders. `vfiBusy` only guards the double
  // click between the POST and that first poll.
  const [vfiBusy, setVfiBusy] = useState(null);
  // ↗ The finished clip the Smooth window was opened for, or null. The rate
  // is asked there (×2, ×3, ×4 of the source), never assumed.
  const [vfiClip, setVfiClip] = useState(null);
  // ✨ Neural render. `nrClip` is the finished clip the dialog was opened
  // for; the render itself is a queued row like any other, so the list's
  // poll shows it land and `nrBusy` only guards the double click.
  const [nrClip, setNrClip] = useState(null);
  const [nrBusy, setNrBusy] = useState(null);
  const [continueBusy, setContinueBusy] = useState(null);
  // ⇔ The rendered clip being compared with its source, or null.
  const [compareClip, setCompareClip] = useState(null);
  // ✨ The Motion helpers. `motionBusy` names WHICH one is running so the two
  // buttons cannot both spin, and the enhancer toggle is a per-run choice —
  // remembered nowhere, because it changes what the sampler reads.
  const [motionBusy, setMotionBusy] = useState(null);
  const [enhanceOn, setEnhanceOn] = useState(false);
  // ⚙ The model window, and the model it settled on — kept here so both
  // buttons send it without re-reading a setting on every click.
  const [modelOpen, setModelOpen] = useState(false);
  const [motionModel, setMotionModel] = useState('');
  const neuralRender = async (clip, params) => {
    setNrBusy(clip.id);
    try {
      await postJson(clipNeuralRenderUrl(clip.id), params);
      setNrClip(null);
      toast.info?.('Neural render queued — the new clip appears below when it is done.');
      await refreshClips();
    } catch (e) {
      toast.error(e?.message || 'That clip could not be neural-rendered.');
    } finally {
      setNrBusy(null);
    }
  };
  const smooth = async (clip, multiplier) => {
    setVfiBusy(clip.id);
    try {
      const r = await postJson(clipVfiUrl(clip.id), { multiplier });
      setVfiClip(null);
      toast.info?.(`Smoothing to ${Math.round(r?.fps || 0) || '…'} fps queued — the new clip appears below when it is done.`);
      await refreshClips();
    } catch (e) {
      toast.error(e?.message || 'That clip could not be smoothed.');
    } finally {
      setVfiBusy(null);
    }
  };

  /* The clip length the dials are set to, as the readback shows it — and as
     the ✨ writers receive it. A 1 s clip and a 15 s clip are not the same
     clip, and a writer that does not know which it is writing paces both the
     same way; this is the value the whole Motion field is timed against. */
  const fps = options?.fps || 24;
  const seconds = clipSeconds(opts.frames, fps);

  /* ✨ Propose the movement from the staged start frame. A PROPOSAL: the model
     sees a still, so it can read who is there and how they are posed, never
     what happens next — the button says "Auto", the note says where it came
     from, and the text stays editable like anything typed by hand. */
  // The local-LLM fence, the image studio's way: a refusal is not an error to
  // toast, the notice takes over and replays the click when the model frees
  // up — or offers the unload. A replay fails outside the try/catch below, so
  // it gets its own voice.
  const { fence, runGuarded, unloadAndRetry, stopWaiting } = useOllamaFence({
    onError: (e) => toast.error(said(e, 'The motion writer could not answer.')),
  });
  // A click made for one mode, one frame or one length must not replay for
  // another: the guard keeps the ACTION, with the frame, the mode and the
  // length it was clicked under, and a switch while it waits would write
  // that answer — a motion paced for the old length — into the new setup.
  useEffect(() => { stopWaiting(); }, [mode, source.image, seconds, stopWaiting]);
  // And a switch while it RUNS: the request is in flight, the guard cannot
  // stop it, and its answer would land in the new setup all the same — so
  // each action asks the guard before writing, and says so when told no
  // (nothing else shows it: the notice is gone, the field unchanged).
  const setAside = () => toast.info(SUPERSEDED_ANSWER_NOTICE);

  const autoMotion = async () => {
    if (!source.image) { toast.warning('Pick a start frame first.'); return; }
    setMotionBusy('auto');
    // The action, not the click: the guard keeps it and replays it verbatim,
    // so the frame, the instruction and the length are captured here.
    const suggest = async (run) => {
      // What is already written STEERS the proposal instead of being replaced
      // by it: the frame says what is there, this says what should happen in it.
      const r = await postJson(motionSuggestUrl(),
        { image: source.image, instruction: prompt, model: motionModel, seconds });
      // Nothing came back: nothing to write, nothing to set aside — the
      // notice is for an answer, not for an empty reply.
      if (r?.prompt && keepAnswer(run, setAside)) setPrompt(r.prompt);
    };
    try {
      await runGuarded(suggest);
    } catch (e) {
      toast.error(said(e, 'The motion could not be written.'));
    } finally {
      setMotionBusy(null);
    }
  };

  /* ✨ Enrich what is already there. Never destructive: the field is written
     only when an answer came back, and a model that answered nothing usable is
     an error with its sentence — a click can cost time, never the sentence
     somebody wrote. */
  const enhanceMotion = async () => {
    setMotionBusy('enhance');
    const enrich = async (run) => {
      // The start frame travels too: an enrichment anchored on the picture
      // that will actually be animated cannot add scenery the frame lacks.
      const r = await postJson(motionEnhanceUrl(),
        { prompt, image: mode === 't2v' ? null : (source.image || null),
          model: motionModel, seconds });
      if (!keepAnswer(run, setAside)) return;
      // "Nothing to add" and "it worked" look the same in the field; the
      // server says which, so a silent click is never mistaken for a rewrite.
      if (r?.unchanged) toast.info('The model had nothing to add — your text is unchanged.');
      else if (r?.prompt) setPrompt(r.prompt);
    };
    try {
      await runGuarded(enrich);
    } catch (e) {
      toast.error(said(e, 'The motion could not be enriched.'));
    } finally {
      setMotionBusy(null);
    }
  };

  const reuse = (clip) => {
    setPrompt(clip.prompt || '');
    setMode(clip.mode === 't2v' ? 't2v' : 'i2v');
    setAspect(clip.aspect || 'auto');
    setOpts({
      accel: clipAccel(clip), eros: !!clip.eros, sparse: clip.sparse || '',
      latentUpscale: !!clip.latent_upscale,
      // A joined clip's `frames` is the FILE's count (parent + part − 1), not
      // a count the sampler takes: reused, it would read "723 frames" and
      // render 362. The dial keeps its value; every other dial is replayed.
      frames: clip.joined ? opts.frames : (clip.frames || opts.frames),
      megapixels: clip.megapixels || opts.megapixels,
      // Reuse replays the count the clip ACTUALLY ran, never "auto" — the
      // whole point of ↻ Reuse is that the second run is the first one with
      // one dial moved.
      steps: clip.steps || '',
      seed: clip.seed ?? '',
    });
    if (clip.lora) {
      setLora({ lora: clip.lora, runId: clip.run_id, datasetId: clip.dataset_id });
      setStrength(clip.lora_strength ?? 1);
    }
    // The start frame comes back too, or Reuse restores every dial except the
    // one that decides whether Generate works: an image-to-video clip reused
    // without its frame lands blocked on "Pick a start frame". The staged file
    // is still in ComfyUI's input folder — the name is all the graph needs, and
    // the server re-reads the shape from the file when it is not sent.
    if (clip.mode !== 't2v' && clip.source_image) {
      // The one frame this clip came from, alone in the strip: Reuse means
      // "this clip again, one thing changed", not "this clip and the batch".
      // The frames it replaces let go of their upload previews first.
      sources.forEach(releasePreview);
      setSources([{ key: `staged:${clip.source_image}`, image: clip.source_image, ratio: null, preview: null }]);
    }
    toast.info?.('Settings loaded — change one thing and generate again.');
  };

  const needsImage = mode === 'i2v' && sources.length === 0;
  // ✨ Written per picture needs no typed motion: an empty field asks the
  // writer for a proposal from each picture alone — a gate on the field
  // refused exactly the case the mode promises (found in verification).
  const perPictureReady = mode === 'i2v' && promptMode === 'per-image' && sources.length > 1;
  const blocked = busy || needsImage || (!prompt.trim() && !perPictureReady);
  const reason = needsImage
    ? 'Pick a start frame, or switch to text-only.'
    : (!prompt.trim() && !perPictureReady ? 'Describe the motion first.' : null);

  /* The readback: what is about to be rendered, in one line, next to the
     button — the moment before a multi-minute job is the moment to catch
     "wrong LoRA" or "still on 10Eros". */
  const readback = [
    lora.lora ? `${shortLoraName(lora.lora)} @ ${Number(strength).toFixed(2)}` : 'no LoRA',
    mode === 't2v' ? 'text only' : (sources.length > 1 ? `from ${sources.length} images` : 'from an image'),
    seconds ? `${seconds}s` : `${opts.frames} frames`,
    `${Number(opts.megapixels).toFixed(2)} MP`,
    opts.accel ? (opts.accel === 'turbo' ? 'turbo' : accelLabel(opts.accel)) : null,
    opts.eros ? '10Eros' : null,
    opts.sparse ? `sparse ${opts.sparse}` : null,
    opts.latentUpscale ? 'upscale ×2' : null,
    // Only when it was CHOSEN: "auto" belongs in the dial's own label, and a
    // readback that always claimed a step count would make the automatic case
    // look like a decision somebody made.
    opts.steps ? `${opts.steps} steps` : null,
  ].filter(Boolean).join(' · ');

  // The button counts what a click queues ("Generate 3 clips"), and where the
  // walk is while it queues — the same text in the rail and in the phone's
  // bar, which is handed the running text too (its own convention is a bare
  // "…" while a run is on; a batch has a count to show).
  const label = generateLabel({ mode, count: sources.length, busy, done: progress.done, total: progress.total,
    phase: progress.phase });
  /* ⏭ Continue: the clip's last frame staged as the next start frame — the
     picture is exactly where that clip ended — and the launch marked so the
     render lands joined behind it. The motion is yours to write again. */
  const continueFrom = async (clip) => {
    // Already staged: a second click would stage a second PNG the strip's
    // dedupe then drops — under a success toast.
    if (sources.some((f) => f.key === `continue:${clip.id}`)) {
      toast.info(`Clip #${clip.id} is already in the strip — its last frame is queued to be continued.`);
      return;
    }
    setContinueBusy(clip.id);
    try {
      const r = await postJson(clipLastFrameUrl(clip.id), {});
      setMode('i2v');
      addSources([{ key: `continue:${clip.id}`, image: r.image, ratio: r.ratio,
        preview: clipLastFramePngUrl(clip.id), continues: clip.id }]);
      toast.success(`Last frame of clip #${clip.id} staged — write the next motion, then Generate. The result plays as clip #${clip.id} followed by the new one.`);
      const el = document.getElementById('vs-motion');
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (e) {
      toast.error(e?.message || 'The last frame could not be read.');
    } finally {
      setContinueBusy(null);
    }
  };
  const continuing = sources.filter((f) => f.continues);
  const generateButton = (
    <button type="button" onClick={generate} disabled={blocked}
      className="flex w-full items-center justify-center gap-2 rounded-lg bg-gradient-primary px-4 py-2 text-sm font-semibold text-gray-950 disabled:opacity-40 min-h-10">
      <Play aria-hidden="true" className="h-4 w-4" />
      {label}
    </button>
  );

  // ⏱ Phrased once, from what the server sent; null when it sent nothing.
  const launchAdvice = launchAdviceLines(options?.launch_advice);

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
          One clip per start frame — compare in time, same seed, one dial changed.
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

      {/* How ComfyUI was STARTED decides more than any dial below. The server
          asks the running instance for its argv and its RAM and answers only
          when it can tell (see video_test_studio.launch_advice); measured on a
          48 GB machine, the flag is the difference between 5 minutes and 25
          seconds per clip, so it is said here, before the first launch. */}
      {launchAdvice && (
        <div className="rounded-xl border border-amber-400/40 bg-amber-400/10 p-3 text-sm"
          data-testid="video-launch-advice">
          <p className="font-semibold text-amber-200">{launchAdvice.title}</p>
          <p className="mt-1 text-content-muted">
            The video weights (about {options.launch_advice.weights_gb} GB) are then kept in
            system RAM, and the machine running ComfyUI has {options.launch_advice.ram_total_gb} GB
            — clips take minutes instead of seconds while models page in and out.{' '}
            {launchAdvice.action}
          </p>
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
            <VideoSourcePicker mode={mode} onMode={setMode} frames={sources}
              aspect={aspect} onAspect={setAspect}
              onAdd={addSources} onRemove={removeSource} onClear={clearSources} />
            {mode === 'i2v' && continuing.length > 0 && (
              <p className="rounded-lg border border-border bg-surface-raised px-2.5 py-1.5 text-[0.6875rem] text-content-muted">
                ⏭ {continuing.map((f) => `clip #${f.continues}`).join(', ')}: the render lands joined behind it —
                one video, that clip then the new motion. Remove the frame from the strip to launch a plain clip instead.
              </p>
            )}
          </div>

          <div id="vs-motion" className="flex flex-col gap-1.5 rounded-xl border border-border bg-surface p-3 scroll-mt-16">
            <span className="flex flex-wrap items-center gap-1.5">
              <label htmlFor="vs-motion-text" className="text-sm font-semibold text-content">Motion</label>
              <HelpBadge topic="video-studio-motion-writer" />
              {/* ✨ Auto writes it from the start frame; ✨ Enrich rewrites what
                  is there. Both put their answer in the field and stop — the
                  render is still the user's click, and the text is still
                  theirs to edit. Auto needs a frame; without one it says so
                  rather than proposing a movement for no picture. */}
              <button type="button" onClick={autoMotion}
                disabled={!!motionBusy || mode === 't2v' || !source.image}
                title={mode === 't2v'
                  ? 'Auto reads the start frame — switch to “From an image” to use it'
                  : (source.image ? 'Write the movement from the start frame'
                    : 'Pick a start frame first')}
                className="ml-auto min-h-10 rounded-lg border border-border px-2 py-1 text-[0.6875rem] text-content-muted hover:text-content disabled:opacity-40 lg:min-h-0">
                {motionBusy === 'auto' ? '…' : '✨ Auto'}
              </button>
              <button type="button" onClick={enhanceMotion}
                disabled={!!motionBusy || !prompt.trim()}
                title="Rewrite what is written with more of the detail a sampler can use"
                className="min-h-10 rounded-lg border border-border px-2 py-1 text-[0.6875rem] text-content-muted hover:text-content disabled:opacity-40 lg:min-h-0">
                {motionBusy === 'enhance' ? '…' : '✨ Enrich'}
              </button>
              {/* ⚙ opens the model window — the list belongs at the moment
                  somebody wonders about it, not permanently beside the two
                  buttons that use it. */}
              <button type="button" onClick={() => setModelOpen(true)}
                title="Which model writes the motion"
                aria-label="Which model writes the motion"
                className="min-h-10 rounded-lg border border-border px-2 py-1 text-[0.6875rem] text-content-muted hover:text-content lg:min-h-0">
                ⚙
              </button>
            </span>
            <textarea id="vs-motion-text" value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={5}
              placeholder="What happens in the shot — she turns her head and smiles, the camera pushes in slowly…"
              className="w-full resize-y rounded-lg border border-border bg-app px-2.5 py-2 text-sm text-content" />
            <span className="text-[0.6875rem] text-content-subtle">
              Describe the movement, not the picture: the start frame already says
              what the scene looks like. ✨ Auto and ✨ Enrich answer in H3’s own
              three-field prompt, paced to the clip length you set.
            </span>
            {/* The presets, under the field they write into. They APPEND, like
                ✨ Enrich leaves your text alone — so the picker can be used on a
                half-written prompt without eating it. */}
            <VideoQuickPrompts mode={mode}
              onAppend={(text) => setPrompt((p) => appendQuickPrompt(p, text))} />
            <OllamaFenceNotice fence={fence} onUnload={unloadAndRetry} onStop={stopWaiting} />
            {/* The toggle enriches AT LAUNCH — what runs is what the clip
                records, so a card always names the prompt that really made it.
                Off by default: it changes what the sampler reads. */}
            <label className="flex items-start gap-2 rounded-lg border border-border bg-surface-raised px-2 py-1.5 text-[0.6875rem] text-content-muted">
              <input type="checkbox" checked={enhanceOn} className="mt-0.5"
                onChange={(e) => setEnhanceOn(e.target.checked)} />
              <span className="min-w-0">
                <span className="font-semibold text-content">✨ Enrich at launch</span>
                <span className="block">
                  Rewrites the motion with more detail when you press Generate,
                  and the clip records what actually ran — your field is left as
                  you typed it.
                </span>
              </span>
            </label>
            {/* The batch's prompt, asked only when there IS a batch: two
                choices, so a segmented pair rather than a select. */}
            {mode === 'i2v' && sources.length > 1 && (
              <div data-testid="video-prompt-mode" className="flex flex-col gap-1 rounded-lg border border-border bg-surface-raised px-2 py-1.5 text-[0.6875rem]">
                <span className="font-semibold text-content">Prompt for the {sources.length} pictures</span>
                <div role="radiogroup" aria-label="Prompt for the batch" className="grid grid-cols-2 gap-1 rounded-lg border border-border bg-surface p-0.5">
                  {[['same', 'Same for all'], ['per-image', '✨ Written per picture']].map(([id, text]) => (
                    <button key={id} type="button" role="radio" aria-checked={promptMode === id}
                      onClick={() => setPromptMode(id)}
                      className={`min-h-10 rounded-md px-2 py-1 text-xs font-semibold lg:min-h-0 ${
                        promptMode === id ? 'bg-primary text-white' : 'text-content-muted hover:text-content'}`}>
                      {text}
                    </button>
                  ))}
                </div>
                <span className="text-content-muted">
                  {promptMode === 'per-image'
                    ? 'Before anything is queued, ✨ reads each picture and writes its prompt: your motion enriched with it, or a proposal from the picture alone when the field is empty. One short call per picture, while ComfyUI is idle.'
                    : 'Every clip runs the motion above, on one seed: the clips differ by their picture and nothing else.'}
                </span>
              </div>
            )}
          </div>
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
        <VideoClipHistory clips={clips} onRate={rate} onDelete={remove} onReuse={reuse} onVfi={setVfiClip} vfiBusy={vfiBusy}
          onNeuralRender={(clip) => setNrClip(clip)} nrBusy={nrBusy}
          onCompare={(clip) => setCompareClip(clip)}
          onJumpTo={jumpTo} onContinue={continueFrom} continueBusy={continueBusy}
          hasMore={hasMore} loadingMore={loadingMore} onLoadMore={loadMore} />
      </section>

      <StudioActionBar shortcuts={SHORTCUTS} canRun={!blocked} running={busy}
        onRun={generate} runLabel={`▶ ${label}`} runningLabel={`▶ ${label}`} note={reason} />

      {/* ↗ The rate Smooth makes, asked before it runs: 48, 72 or 96 fps for
          a 24 fps clip — the interpolator works by whole factors. */}
      {vfiClip && (
        <SmoothDialog clip={vfiClip} busy={vfiBusy === vfiClip.id}
          onSmooth={(multiplier) => smooth(vfiClip, multiplier)}
          onClose={() => setVfiClip(null)} />
      )}
      {/* ✨ The neural render dials, asked once per clip. The capability's own
          sentences come with the options payload, so the dialog can refuse
          in words on a machine without the model. */}
      {nrClip && (
        <NeuralRenderDialog status={options?.neural_render} busy={nrBusy === nrClip.id}
          initial={nrClip.nr_params || null}
          subject={`Clip #${nrClip.id}${nrClip.seconds ? ` (${nrClip.seconds}s)` : ''}.`}
          consequence="The render is a NEW clip in this list; the original stays as it is."
          onRender={(params) => neuralRender(nrClip, params)}
          onClose={() => setNrClip(null)} />
      )}
      {/* ⇔ Source and render side by side, in step. The source is the row the
          render points at; if it was deleted, the left side says so. */}
      {compareClip && (
        <SideBySideVideo originalSrc={clipVideoUrl(compareClip.nr_of)}
          renderSrc={clipVideoUrl(compareClip.id)}
          title={`clip #${compareClip.nr_of} → neural render #${compareClip.id}`}
          exportHref={clipComparisonUrl(compareClip.id)}
          onClose={() => setCompareClip(null)} />
      )}
      {/* ⚙ The model that writes the motion, on demand. */}
      {modelOpen && (
        <MotionModelDialog onClose={() => setModelOpen(false)}
          onSaved={setMotionModel} />
      )}
    </div>
  );
}
