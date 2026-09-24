import { PluginSlot, hasContributions } from '@lds/plugin-sdk/ui'
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
import { apiFetch, del, postJson } from '@lds/plugin-sdk';
import { HelpBadge } from '@lds/plugin-sdk';
import { useOllamaFence } from '@lds/plugin-sdk/inference';
import { SUPERSEDED_ANSWER_NOTICE, keepAnswer } from '@lds/plugin-sdk/inference';
import { OllamaFenceNotice } from '@lds/plugin-sdk/inference';
import { useToast } from '@lds/plugin-sdk';
import { StudioActionBar } from '@lds/plugin-sdk/ui';
import VideoClipHistory from './VideoClipHistory';
import VideoBestSettings, { useVideoBestSettings } from './VideoBestSettings.jsx';
/* The extension is NOT decoration here: `VideoBestSettings.jsx` and
   `videoBestSettings.js` differ only by case, so a bare specifier resolved
   with .jsx first lands on the component on Windows and macOS. That is what
   kept this file out of the mount harness — and out of the one test that
   would have caught a white screen (found in verification, 2026-09-07). */
import { bestToControls, bestUnavailable } from './videoBestSettings.js';
import AutoContinuePanel from './AutoContinuePanel.jsx';
import useAutoContinue from './useAutoContinue';
import { autoIsBusy, autoLimit, canAutoContinue } from './videoAutoContinue';
import VideoLoraPicker from './VideoLoraPicker.jsx';
import VideoOptionsPanel from './VideoOptionsPanel.jsx';
import VideoQuickPrompts from './VideoQuickPrompts.jsx';
import { appendQuickPrompt } from './videoPromptPresets';
import MotionModelDialog from './MotionModelDialog.jsx';
import SmoothDialog from './SmoothDialog.jsx';
import VideoSourcePicker from './VideoSourcePicker.jsx';
import VideoReferencesPanel from './VideoReferencesPanel.jsx';
import useVideoReferences from './useVideoReferences';
import { ASPECTS, readReferenceDraft, referenceDescriptors, referenceFormat, referenceFormatSize, referencePayload, referenceSummary } from './videoReferences';
import { continuationState, continuesAsReference } from './videoContinuation';
import VideoContinuationNotice from './VideoContinuationNotice.jsx';
import { readPromptDraft, writePromptDraft } from './videoPromptDraft';
import { GeneratedImageLightbox } from '@lds/plugin-sdk/ui';
import { useCanvasImageImprove } from '@lds/plugin-sdk/inference';
import { useRestoreImproveSettings } from '@lds/plugin-sdk/inference';
import { canImproveCanvasImage } from '@lds/plugin-sdk/inference';
import { shortLoraName } from '@lds/plugin-sdk/h3';
import {
  addFrames, failureNotice, generateLabel, perImagePrompts, queueClips, queuedNotice, releasePreview,
  removeFrame,
} from './videoStartFrames';
import {
  accelLabel, clipAccel, clipLastFramePngUrl, clipLastFrameUrl, clipRateUrl, clipSeconds, clipUrl, clipsUrl,
  buildGeneratePayload, generateUrl, mergeClipPages,
  pickAvailableAccel,
  isRunning, launchAdviceLines, optionsUrl, clipVfiUrl, clipNeuralRenderUrl, clipVideoUrl,
  clipComparisonUrl,
  motionEnhanceUrl, motionSuggestUrl, motionWriteBatchUrl, SHOT_CHOICES, shotCap,
  frameAdoptBody, frameAdoptUrl, frameDatasetIdOf, sourceUrl, sparseInForce, writerContext,
} from './videoStudioApi';

/* No start frame yet — what the ✨ helpers and the readback see before a pick. */
import { PERFORMANCE_DEFAULTS, performanceSettings, referenceBaseMissing } from './videoPerformance.js';

const EMPTY_SOURCE = { image: null, ratio: null, preview: null };

/* An acceleration ON by default — larryvrh's, the arena's first row. Without
   one the base is undistilled and a first clip is tens of minutes — long
   enough that a new user concludes the studio is broken rather than slow. The
   panel says what each choice changes. */
const DEFAULT_OPTIONS = {
  ...PERFORMANCE_DEFAULTS,
  accel: 'turbo', eros: false, light: false, shots: 1, sparse: '', latentUpscale: false,
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

/* `datasetId` is the image dataset the Studio was opened FROM (StudioPage
   resolves `/dataset/studio/:id` and `/studio?dataset=`), or null on the
   plain /studio the nav opens — the video lane sits under no dataset. It
   only decides where an opened start frame gets its library row. */
export default function VideoTestStudio({ datasetId = null } = {}) {
  const toast = useToast();
  const [options, setOptions] = useState(null);
  const [lora, setLora] = useState({ lora: null, runId: null, datasetId: null });
  const [strength, setStrength] = useState(1.3);
  const [mode, setMode] = useState(() => readReferenceDraft().active ? 'ref2va' : 'i2v');
  const [aspect, setAspect] = useState('landscape');
  // The start frames, in pick order: the strip the picker draws and the list
  // Generate walks — one clip each, on one seed (one frame was the whole state
  // until 2026-09-02). `source` is the FIRST of them: what ✨ Auto and ✨ Enrich
  // read, and what a change of resets the poller on.
  const [sources, setSources] = useState([]);
  const source = sources[0] || EMPTY_SOURCE;
  // ⏭ The frame the ✨ writers take their chain from: the one that continues a
  // clip when the strip holds one — ⏭ Continue APPENDS to the strip, so the
  // first frame is usually an older pick (found in verification, 2026-09-04).
  const chainFrame = sources.find((f) => f.continues) || source;
  const addSources = useCallback((list) => setSources((prev) => addFrames(prev, list).frames), []);
  const removeSource = useCallback((key) => setSources((prev) => removeFrame(prev, key)), []);
  const clearSources = useCallback(() => setSources([]), []);
  // 🎞 The picture the clip ENDS on — one per launch, shared by every clip of
  // a batch (H3 first-last conditioning). null = a free ending.
  const [endFrame, setEndFrame] = useState(null);
  // How far a batch is between the click and the last reply, for the button.
  const [progress, setProgress] = useState({ done: 0, total: 0, phase: 'queueing' });
  /* The batch's prompt: ONE for every picture (the default, the comparison
     that says something about the LoRA), or one WRITTEN per picture by ✨ —
     the frame read by the vision model, the typed motion enriched with it or
     a proposal from the picture alone. Written before anything is queued:
     the writer's window shuts once a clip sits in the queue. */
  const [promptMode, setPromptMode] = useState('same');
  // Kept in this browser like the references (2026-09-06): a reload, or a
  // trip to another page, gives the field back as it was typed.
  const [prompt, setPrompt] = useState(() => readPromptDraft());
  useEffect(() => { writePromptDraft(prompt); }, [prompt]);
  const [opts, setOpts] = useState(DEFAULT_OPTIONS);
  const reference = useVideoReferences(setPrompt);
  const isReference = mode === 'ref2va';
  const useRefmods = !isReference && reference.references.length > 0;
  const launchMode = mode === 'i2v' && !sources.length && useRefmods ? 't2v' : mode;
  const formatReference = referenceFormat(reference.references);
  const renderOpts = isReference ? { ...opts, ...reference.settings,
    eros: false, light: false, latentUpscale: false,
    sparse: options?.options_available?.sparse?.available === false ? '' : reference.settings.sparse } : opts;
  const writerFrame = isReference ? reference.firstFrame : chainFrame;
  const writerEnd = isReference ? reference.endFrame : endFrame;
  const [writerNotice, setWriterNotice] = useState('');
  const updateReference = reference.update;
  useEffect(() => { updateReference({ active: mode === 'ref2va' }); }, [mode, updateReference]);
  useEffect(() => { setWriterNotice(''); }, [mode, reference.signature]);
  const [clips, setClips] = useState([]);
  // ComfyUI's progress bar for the clip on the GPU, as the clips listing
  // carries it — refreshed by the same poll that watches the clip itself.
  const [renderProgress, setRenderProgress] = useState(null);
  const [busy, setBusy] = useState(false);
  const best = useVideoBestSettings(lora, toast);
  const applyBest = (saved) => {
    const error = bestUnavailable(saved, options, mode);
    if (error) { toast.error(error); return; }
    const next = bestToControls(saved, options, opts);
    setMode(next.mode); setAspect(next.aspect); setLora(next.lora);
    setStrength(next.strength); setOpts(next.opts);
    toast.success('Best settings applied. Your motion and frames are unchanged.');
  };
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
        setOpts((o) => ({ ...o, accel: o.fused ? '' : pickAvailableAccel(o.accel, d.accelerations) }));
      }
      if (d?.megapixels?.default) {
        setOpts((o) => ({ ...o, megapixels: d.megapixels.default }));
      }
      // 🪶 The lighter base's verdict arrives with the options: a box ticked
      // before the reply (the fail-open window) must not keep announcing a
      // base the server just said it cannot run.
      if (d?.light?.available === false) setOpts((o) => ({ ...o, light: false }));
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
      setRenderProgress(d.render || null);
      if (!oldestLoadedRef.current || boundary < oldestLoadedRef.current) oldestLoadedRef.current = boundary;
      setHasMore(!!d.has_more);
      return fresh;
    } catch {
      return [];
    }
  }, []);
  const auto = useAutoContinue(refreshClips);
  const [autoDirection, setAutoDirection] = useState('');
  const [autoMaxClips, setAutoMaxClips] = useState('0');
  const autoDraftId = useRef(null);
  useEffect(() => {
    if (auto.session && autoDraftId.current !== auto.session.id) {
      autoDraftId.current = auto.session.id;
      setAutoDirection(auto.session.direction || '');
      setAutoMaxClips(String(auto.session.max_clips || 0));
    }
  }, [auto.session]);
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
    // `prompt` = enrich THIS text on every frame (the typed motion, as the
    // Enrich button does). `instruction` = the same text as a STEER when the
    // batch PROPOSES instead (as the Auto button does): the frame says what is
    // there, the field says what should happen in it. The first port sent only
    // `prompt`, so a proposing batch ignored what the user had typed.
    const reply = await postJson(motionWriteBatchUrl(), {
      images: frames.map((f) => f.image),
      ...(useRefmods ? { refmods: true, references: referencePayload(reference.references) } : {}),
      // ⏭ per picture: a frame staged by Continue names the clip it follows.
      continues: frames.map((f) => f.continues || null),
      // 🎞 one last frame for the whole strip, when one is picked.
      end_image: endFrame?.image || null,
      prompt: (typed && typed.trim()) ? typed : '',
      instruction: (typed && typed.trim()) ? typed : '',
      model: motionModel, seconds, shots: opts.shots,
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
    let launches = isReference ? [reference.firstFrame] : launchMode === 't2v' ? [null] : sources;
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
        mode: launchMode, prompt, aspect: isReference ? reference.settings.aspect : aspect,
        lora: lora.lora, loraStrength: strength, runId: lora.runId,
        datasetId: lora.datasetId, endImage: writerEnd?.image || null, ...renderOpts,
        ...(useRefmods ? { refmods: true, references: referencePayload(reference.references) } : {}),
        ...(isReference ? { references: referencePayload(reference.references),
          refBase: reference.settings.base, refImageSize: reference.settings.imageSize } : {}),
      }, (body) => postJson(generateUrl(), body), (done, total) => setProgress({ done, total, phase: 'queueing' }));
      if (outcome.failed) toast.error(failureNotice(outcome));
      else toast.success(queuedNotice(outcome));
      // The launch went through with the prompt as typed: the writer could
      // not run (fence, server away). Said, or the checkbox looks ignored.
      if (outcome.enrichSkipped) toast.warning(`Queued without enrichment — ${outcome.enrichSkipped}`);
      const warnings = outcome.queued.flatMap((r) => r?.warnings || r?.context_warnings || []);
      if (warnings.length) setWriterNotice([...new Set(warnings)].join(' '));
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
      // ⏭ …and the continuation it was armed for, or the panel keeps promising
      // a join behind a clip that is gone: the guide outlives the card (it is
      // kept in this browser), so the promise survived a reload and the launch
      // was refused minutes later. Done HERE, on the delete that actually went
      // through — never inferred from a clip missing off the list, which is
      // paginated (verification, 2026-09-07). Said, too: the guide picture and
      // the shape lock disappear with it, and a panel that rearranges itself
      // after a click somewhere else owes a sentence.
      if (continuation.used.includes(clip.id) || continuation.ignored.some((c) => c.id === clip.id)) {
        toast.info(`Clip #${clip.id} is gone, so the continuation armed on it was dropped.`);
      }
      disarmContinuation(clip.id);
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
  /* 🔍 The strip's viewer is the SHARED one every surface opens on a picture,
     so a start frame gets the same verbs as the Gallery — ✨ improve, 🔍
     upscale, ✦ repair, 📷 camera angles, 📤 Civitai — rather than a viewer of
     its own with three of them. What the viewer needs is a library row: the
     server gives the frame one (frame/adopt — a content-addressed copy in the
     dataset folder, once), and the row travels with the strip index so ‹ ›
     walk the frames. The row lives in the page's image dataset when the
     Studio was opened from one, else in the server's holding dataset —
     never in the LoRA's, which is a VIDEO dataset (another table). */
  const [zoom, setZoom] = useState(null);
  const frameDatasetId = frameDatasetIdOf({ pageDatasetId: datasetId });
  const restoreImproveSettings = useRestoreImproveSettings();
  const improveImage = useCanvasImageImprove({
    launchMessage: (label) => `${label || 'Improve'} started — the result lands in the 🖼 Gallery; pick it from the Gallery tab to animate it.`,
  });
  const openFrame = useCallback(async (frame, index) => {
    try {
      const r = await postJson(frameAdoptUrl(), frameAdoptBody(frame, frameDatasetId));
      // The row's id rides on the frame: the picker's Gallery tab then reads
      // that picture as already in the strip instead of offering it again.
      setSources((prev) => prev.map((f, i) => (i === index ? { ...f, galleryImageId: r.image.id } : f)));
      setZoom({ index, img: r.image });
    } catch (e) {
      toast.error(e?.message || 'The start frame could not be opened.');
    }
  }, [frameDatasetId, toast]);
  /* 🎞 The last frame opens in the same viewer; its index is -1 so ‹ › do
     not walk the strip from it and a repair re-stages IT, not a start frame. */
  const openEndFrame = useCallback(async (frame) => {
    try {
      const bare = { ...frame, key: String(frame.key || '').replace(/^end:/, '') };
      const r = await postJson(frameAdoptUrl(), frameAdoptBody(bare, frameDatasetId));
      setEndFrame((cur) => (cur ? { ...cur, galleryImageId: r.image.id } : cur));
      setZoom({ index: -1, img: r.image });
    } catch (e) {
      toast.error(e?.message || 'The last frame could not be opened.');
    }
  }, [frameDatasetId, toast]);
  /* ✦ Repair rewrote the library file, and the strip animates a COPY staged
     into ComfyUI: the repaired picture is staged again into the same slot, so
     what the viewer shows is what the next clip starts from. */
  const restageFrame = useCallback(async (index, img) => {
    try {
      const r = await postJson(sourceUrl(), { gallery_image_id: img.id });
      if (index < 0) {
        setEndFrame((cur) => {
          if (!cur) return cur;
          releasePreview(cur);
          return { ...cur, image: r.image, ratio: r.ratio, preview: `${img.url}?v=${Date.now()}` };
        });
        return;
      }
      setSources((prev) => prev.map((f, i) => {
        if (i !== index) return f;
        releasePreview(f);   // an upload's blob: URL is let go before it is replaced
        return { ...f, image: r.image, ratio: r.ratio, preview: `${img.url}?v=${Date.now()}` };
      }));
    } catch (e) {
      toast.error(e?.message || 'The repaired picture could not be staged.');
    }
  }, [toast]);
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
  const seconds = clipSeconds(renderOpts.frames, fps);
  // 🎬 The most shots this length can hold, by the server's rule; a choice
  // that the length then outgrows is brought back to the cap rather than
  // sent as a number the server would trim in silence.
  const shotsCap = shotCap(seconds);
  // The cap in words, singular where it is one — the sentence and the title
  // share it, so neither can say "1 shots".
  const shotsCapNote = shotsCap === 1
    ? `a ${seconds}s clip holds a single shot`
    : `a ${seconds}s clip holds at most ${shotsCap} shots`;
  useEffect(() => {
    if (opts.shots > shotsCap) setOpts((o) => ({ ...o, shots: shotsCap }));
  }, [shotsCap, opts.shots]);

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
  useEffect(() => { stopWaiting(); }, [mode, source.image, seconds, reference.signature, stopWaiting]);
  // And a switch while it RUNS: the request is in flight, the guard cannot
  // stop it, and its answer would land in the new setup all the same — so
  // each action asks the guard before writing, and says so when told no
  // (nothing else shows it: the notice is gone, the field unchanged).
  const setAside = () => toast.info(SUPERSEDED_ANSWER_NOTICE);

  const autoMotion = async () => {
    if ((isReference || useRefmods) ? !reference.references.length : launchMode === 't2v' || !source.image) {
      toast.warning(isReference ? 'Add a reference first.' : 'Pick a start frame first.'); return;
    }
    setMotionBusy('auto');
    // The action, not the click: the guard keeps it and replays it verbatim,
    // so the frame, the instruction and the length are captured here.
    const suggest = async (run) => {
      // What is already written STEERS the proposal instead of being replaced
      // by it: the frame says what is there, this says what should happen in it.
      const r = await postJson(motionSuggestUrl(),
        { image: isReference ? reference.firstFrame?.image : launchMode === 't2v' ? null : source.image,
          instruction: prompt, model: motionModel, seconds, shots: opts.shots,
          ...writerContext({ frame: writerFrame, endFrame: writerEnd, mode: launchMode, refmods: useRefmods,
            references: referencePayload(reference.references), refBase: reference.settings.base,
            refImageSize: reference.settings.imageSize }) });
      // Nothing came back: nothing to write, nothing to set aside — the
      // notice is for an answer, not for an empty reply.
      if (r?.prompt && keepAnswer(run, setAside)) {
        setPrompt(r.prompt); setWriterNotice((r.warnings || []).join(' '));
      }
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
        { prompt, image: isReference ? reference.firstFrame?.image : launchMode === 't2v' ? null : (source.image || null),
          model: motionModel, seconds, shots: opts.shots,
          ...writerContext({ frame: writerFrame, endFrame: writerEnd, mode: launchMode, refmods: useRefmods,
            references: referencePayload(reference.references), refBase: reference.settings.base,
            refImageSize: reference.settings.imageSize }) });
      if (!keepAnswer(run, setAside)) return;
      setWriterNotice((r.warnings || []).join(' '));
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
    setMode(clip.mode === 'ref2va' ? 'ref2va' : clip.mode === 't2v' ? 't2v' : 'i2v');
    setAspect(clip.aspect || 'auto');
    if (clip.mode === 'ref2va') reference.restore(clip);
    else {
      reference.setSettings({ refmods: clip.generation_settings?.refmods === true });
      // Reuse supplies its own prompt, already numbered for this clip's refs.
      reference.setReferences(referenceDescriptors(clip.references), { remapPrompt: false });
    }
    setOpts({
      ...performanceSettings(clip.generation_settings),
      accel: clip.mode === 'ref2va' ? opts.accel : clipAccel(clip), eros: !!clip.eros, light: !!clip.light,
      sparse: clip.mode === 'ref2va' ? opts.sparse : clip.sparse || '',
      latentUpscale: !!clip.latent_upscale,
      // 🎬 A clip records no shot plan — the reused prompt carries its own cut
      // marks — so the row keeps its value. A replacement that dropped the key
      // left the row with nothing selected (found by verification, 2026-09-04).
      shots: opts.shots,
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
    } else setLora({ lora: null, runId: null, datasetId: null });
    // The start frame comes back too, or Reuse restores every dial except the
    // one that decides whether Generate works: an image-to-video clip reused
    // without its frame lands blocked on "Pick a start frame". The name is all
    // the graph needs: the server puts the file back into ComfyUI's input
    // folder from the clip's own copy when the boot sweep has cleared it, and
    // re-reads the shape from it when it is not sent.
    if (clip.mode !== 't2v' && clip.source_image) {
      // The one frame this clip came from, alone in the strip: Reuse means
      // "this clip again, one thing changed", not "this clip and the batch".
      // The frames it replaces let go of their upload previews first.
      sources.forEach(releasePreview);
      // ⏭ …and the clip it continued, so the reused launch continues the same
      // parent (a branch, which the Continue chapter presents as a feature)
      // and the ✨ writers keep their chain.
      setSources([{ key: `staged:${clip.source_image}`, image: clip.source_image, ratio: null, preview: null,
        continues: clip.continues_of || null }]);
    }
    // 🎞 And the frame it ended on, when it had one — a dial of its own,
    // replayed (or cleared) whatever the mode: a text-only clip can end on a
    // picture too, and a last frame left armed from before would otherwise
    // ride into "this clip again" (found in verification, 2026-09-04).
    releasePreview(endFrame);
    setEndFrame(clip.end_image
      ? { key: `end:staged:${clip.end_image}`, image: clip.end_image, ratio: null, preview: null } : null);
    toast.info?.('Settings loaded — change one thing and generate again.');
  };

  const needsImage = mode === 'i2v' && sources.length === 0 && !useRefmods;
  const needsReferences = (isReference || useRefmods) && reference.references.length === 0;
  const invalidRefmods = useRefmods && reference.references.some(r => r.kind !== 'image');
  const referenceProfileMissing = isReference && (
    referenceBaseMissing(options?.reference?.bases?.find((b) => b.id === reference.settings.base), reference.settings, options?.performance)
    || options?.reference?.accelerations?.find((a) => a.id === reference.settings.accel)?.available === false);
  const removedReference = /\[removed (?:Picture|Video|Audio) \d+\]/.exec(prompt);
  const editRemovedReference = () => {
    const field = document.getElementById('vs-motion-text');
    if (!field || !removedReference) return;
    field.focus();
    field.setSelectionRange(removedReference.index, removedReference.index + removedReference[0].length);
    field.scrollIntoView({ behavior: 'smooth', block: 'center' });
  };
  // ✨ Written per picture needs no typed motion: an empty field asks the
  // writer for a proposal from each picture alone — a gate on the field
  // refused exactly the case the mode promises (found in verification).
  const perPictureReady = mode === 'i2v' && promptMode === 'per-image' && sources.length > 1;
  const blocked = invalidRefmods || busy || !!motionBusy || reference.staging || needsImage || needsReferences || referenceProfileMissing || removedReference || (!prompt.trim() && !perPictureReady);
  const reason = invalidRefmods ? 'RefMods accept identity images only. Remove video or audio references.' : needsReferences ? (useRefmods ? 'Add an identity image.' : 'Add an image, video or audio reference.')
    : referenceProfileMissing ? 'Install the selected reference profile from Setup, then refresh this panel.'
      : removedReference ? `The motion contains ${removedReference[0]}. Edit this mention before generating.` : needsImage
    ? 'Pick a start frame, add an identity reference, or switch to text-only.'
    : (!prompt.trim() && !perPictureReady ? 'Describe the motion first.' : null);

  /* ⏭ What THIS launch is joined behind, and what stays armed that it will
     not use. Both are shown: a continuation is a click that means something,
     and a mode that cannot honour it has to say so rather than render a
     stranger under a "Queued" toast.
     Declared HERE, above the readback that reads it: a `const` is in its
     temporal dead zone until its own line, so the same two lines written in
     the other order threw a ReferenceError on EVERY render — a white studio
     in every mode, armed or not (found in verification, 2026-09-07). */
  const continuation = continuationState({ mode, sources, firstFrame: reference.firstFrame });
  // The clip the reference guide continues, when it continues one: the shape
  // control reads it, and so does the panel's own summary.
  const guideContinues = reference.firstFrame?.continues || null;
  const disarmContinuation = (id) => {
    setSources((prev) => prev.filter((f) => f.continues !== id));
    if (reference.firstFrame?.continues === id) reference.update({ firstFrame: null });
  };

  /* The readback: what is about to be rendered, in one line, next to the
     button — the moment before a multi-minute job is the moment to catch
     "wrong LoRA" or "still on 10Eros". */
  const readback = [
    lora.lora ? `${shortLoraName(lora.lora)} @ ${Number(strength).toFixed(2)}` : 'no LoRA',
    isReference ? `references: ${referenceSummary(reference.references) || 'none'}`
      : launchMode === 't2v' ? (useRefmods ? 'prompt + identity references' : 'text only') : (sources.length > 1 ? `from ${sources.length} images` : 'from an image'),
    // ⏭ The join, in the line read the moment before a multi-minute click —
    // and, when one is armed that this mode does not run, that it is NOT one.
    continuation.used.length ? `⏭ joined behind ${continuation.used.map((id) => `#${id}`).join(', ')}`
      : continuation.ignored.length
        ? `⏭ ${continuation.ignored.map(({ id }) => `#${id}`).join(', ')} NOT continued` : null,
    seconds ? `${seconds}s` : `${renderOpts.frames} frames`,
    // 🎬 Only when a writer runs at THIS launch: the plan is read by the
    // enrich-at-launch rewrite and the per-picture batch, never by the sampler.
    (opts.shots > 1 && (enhanceOn || promptMode === 'per-image')) ? `✨ ${opts.shots} shots` : null,
    `${Number(renderOpts.megapixels).toFixed(2)} MP`,
    renderOpts.accel ? (renderOpts.accel === 'turbo' ? 'turbo' : accelLabel(renderOpts.accel)) : null,
    isReference ? `${reference.settings.base} reference base` : opts.eros ? '10Eros' : null,
    renderOpts.light ? 'W4A8 base' : null,
    writerEnd ? 'ends on a picture' : null,
    useRefmods ? `RefMods: ${reference.references.length} identity image${reference.references.length > 1 ? 's' : ''}` : null,
    sparseInForce(renderOpts, isReference) ? `sparse ${sparseInForce(renderOpts, isReference)}` : null,
    renderOpts.latentUpscale ? 'upscale ×2' : null,
    // Only when it was CHOSEN: "auto" belongs in the dial's own label, and a
    // readback that always claimed a step count would make the automatic case
    // look like a decision somebody made.
    renderOpts.steps ? `${renderOpts.steps} steps` : null,
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
  /* ⏭ Arm a References clip's continuation on the guide: the cast comes back
     with it — the references are what hold its characters, and continuing from
     the picture alone drops that identity at the seam — and so does what
     decides the LOOK of the seam: the base, its acceleration, the reference
     size and the shape. The dials that pace the new part (length, megapixels,
     seed) stay yours. A selection made for the next shot stays selected;
     the parent's cast is restored only when there is no current selection.
     One function for the full click and for the repair below: the repair used
     to arm the guide WITHOUT the cast, which is the very loss this feature
     exists to prevent (verification, 2026-09-07). */
  const armAsReference = (clip, image, { fresh = true } = {}) => {
    if (!reference.references.length) reference.setReferences(referenceDescriptors(clip.references));
    reference.update({ active: true,
      settings: { ...reference.settings, ...performanceSettings(clip.generation_settings), base: clip.ref_base || 'official',
        accel: clip.accel || '', imageSize: clip.ref_image_size || 'match',
        // 'auto' is what a row that predates the shape carries, and it is not
        // one of the three the select offers: an unknown shape keeps the one
        // in force rather than emptying the control.
        aspect: ASPECTS.includes(clip.aspect) ? clip.aspect : reference.settings.aspect },
      firstFrame: { image, continues: clip.id },
      // The parent's ENDING must not ride into the part: it would pin the new
      // clip to the picture the parent already ended on. Only on a FRESH ⏭
      // though — a repair puts back what was lost, and clearing a last frame
      // the author picked in the meantime would be a silent deletion
      // (verification, 2026-09-07).
      ...(fresh ? { endFrame: null } : {}) });
    setMode('ref2va');
  };

  const continueFrom = async (clip) => {
    /* Already armed, in one place or in both. Read by what a frame CONTINUES,
       not by the key ⏭ gives it: ↻ Reuse hands the same continuation back
       under `staged:…`, and a key test let a second copy of the same picture
       through — two identical multi-minute renders on one click (verification,
       2026-09-07).
       And when only ONE side holds it, this REPAIRS instead of refusing: the
       two remove buttons each empty their own side, and a ⏭ that answered
       "already armed" to the state it should fix made itself a no-op. The
       picture is already staged, so re-arming costs no request. */
    if (clip.mode !== 'ref2va') {
      reference.setSettings({ refmods: clip.generation_settings?.refmods === true });
      // Continue uses the selection for the NEXT shot. An older clip often has
      // no saved refs; clearing the new selection also broke its prompt tags.
      if (!reference.references.length) reference.setReferences(referenceDescriptors(clip.references));
      setOpts(o => ({ ...o, ...performanceSettings(clip.generation_settings), accel: clipAccel(clip), steps: clipAccel(clip) === 'taomate_3step' ? 3 : '', eros: !!clip.eros, light: !!clip.light }));
      setMode('i2v');
    }
    const inStrip = sources.find((f) => f.continues === clip.id);
    const asGuide = reference.firstFrame?.continues === clip.id;
    const wantsGuide = continuesAsReference(clip);
    if (inStrip && (asGuide || !wantsGuide)) {
      toast.info(`Clip #${clip.id} is already armed — its last frame is queued to be continued.`);
      return;
    }
    if (inStrip && wantsGuide) {
      armAsReference(clip, inStrip.image, { fresh: false });
      toast.success(`Clip #${clip.id} is armed again with its references, on its last frame.`);
      return;
    }
    if (asGuide) {
      /* Armed on the guide alone. For a take that continues WITH its cast
         there is nothing to repair — the guide IS the right place — so this
         only brings the mode back. Copying it into the strip instead, as this
         branch first did, armed an image continuation of a References take:
         launched from "From an image" it joined behind the parent WITHOUT its
         references, and the amber warning that had been saying so disappeared
         (verification, 2026-09-07). */
      if (wantsGuide) {
        setMode('ref2va');
        toast.info(`Clip #${clip.id} is armed on the first frame guide — back to References to continue it.`);
        return;
      }
      addSources([{ key: `continue:${clip.id}`, image: reference.firstFrame.image, ratio: null,
        preview: clipLastFramePngUrl(clip.id), continues: clip.id }]);
      toast.success(`Clip #${clip.id} is armed again in the start frames.`);
      return;
    }
    setContinueBusy(clip.id);
    const asReference = continuesAsReference(clip);
    try {
      const r = await postJson(clipLastFrameUrl(clip.id), {});
      // ONE staged picture, armed in BOTH places a launch can start from: the
      // strip "From an image" walks, and the guide a reference launch pins at
      // frame 0. The mode picked below decides which one runs — and a change
      // of mode afterwards changes HOW the clip is continued instead of
      // dropping the continuation (measured 2026-09-07: armed in the strip
      // alone, a click back on References hid it and Generate rendered a
      // fresh clip in silence).
      addSources([{ key: `continue:${clip.id}`, image: r.image, ratio: r.ratio,
        preview: clipLastFramePngUrl(clip.id), continues: clip.id }]);
      if (asReference) {
        armAsReference(clip, r.image);
      } else {
        // The BASE follows the clip being continued, whatever box is ticked
        // right now: the joined render starts from that clip's last frame, and
        // a change of base at the seam would show. The dials stay yours.
        setOpts((o) => ({ ...o, ...performanceSettings(clip.generation_settings), eros: !!clip.eros, light: !!clip.light }));
      }
      setMode(asReference ? 'ref2va' : 'i2v');
      toast.success(asReference
        ? `Last frame of clip #${clip.id} staged as the first frame guide, with the selected references — write the next motion, then Generate. The result plays as clip #${clip.id} followed by the new one.`
        : `Last frame of clip #${clip.id} staged — write the next motion, then Generate. The result plays as clip #${clip.id} followed by the new one.`);
      const el = document.getElementById('vs-motion');
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (e) {
      toast.error(e?.message || 'The last frame could not be read.');
    } finally {
      setContinueBusy(null);
    }
  };
  const applyAuto = () => auto.act('update', { session_id: auto.session?.id,
    direction: autoDirection, max_clips: autoLimit(autoMaxClips) });
  const stopAuto = () => auto.act('stop', { session_id: auto.session?.id });
  const resumeAuto = async () => {
    if (await applyAuto()) await auto.act('resume', { session_id: auto.session?.id });
  };
  const toggleAuto = async (clip, checked) => {
    if (!checked) { await stopAuto(); return; }
    if (!canAutoContinue(clip, mode)) {
      toast.info('Auto continuation uses image-to-video clips. Switch to From an image to start a loop.'); return;
    }
    const limit = autoLimit(autoMaxClips);
    if (limit === null) { toast.error('Choose a whole number of clips, or 0 for no limit.'); return; }
    const settings = buildGeneratePayload({ mode: 'i2v', prompt: '',
      lora: lora.lora, loraStrength: strength, runId: lora.runId,
      datasetId: lora.datasetId, ...opts });
    const started = await auto.act('start', { clip_id: clip.id, direction: autoDirection,
      max_clips: limit, model: motionModel || null, settings });
    if (started) toast.success(`Auto continuation started from clip #${clip.id}.`);
  };
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
          {isReference ? 'One clip from your references — keep identities, motion and sound together.'
            : 'One clip per start frame — compare in time, same seed, one dial changed.'}
        </span>
      </header>

      {/* The one banner that has to come BEFORE everything else: on a machine
          without the weights, every control below is a promise the lane cannot
          keep. It names the missing files by what they do and points at the one
          screen that installs them. */}
      {!isReference && options && options.ready === false && (
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
            {isReference ? <>
              <div className="mb-2 flex flex-wrap items-center gap-2 rounded-lg border border-border bg-surface p-2">
                <div role="group" aria-label="Video input mode" className="flex w-full gap-1">
                  {[['i2v', 'From an image'], ['t2v', 'Text only'], ['ref2va', 'References']].map(([id, text]) => (
                    <button key={id} type="button" aria-pressed={mode === id} disabled={busy || reference.staging} onClick={() => setMode(id)}
                      className={`min-h-10 flex-1 rounded-md px-2 py-1 text-xs lg:min-h-0 ${mode === id ? 'bg-primary text-white' : 'text-content-muted hover:text-content'}`}>{text}</button>
                  ))}
                </div>
                {/* ⏭ A continuation has no shape of its own: it renders at its
                    parent's, which the server takes from the seam picture. The
                    control says so and stops offering a choice that would be
                    overridden — the same shape the format of a reference video
                    already gives it. */}
                <label className="flex w-full items-center gap-2 text-xs text-content-muted">Output shape
                  <select value={guideContinues ? 'continuation' : formatReference ? 'reference' : reference.settings.aspect}
                    disabled={!!formatReference || !!guideContinues}
                    title={guideContinues ? `The part follows clip #${guideContinues}, so the join has nothing to rescale.`
                      : formatReference ? 'Uncheck the video’s format option to choose a shape.' : undefined}
                    onChange={(e) => reference.setSettings({ aspect: e.target.value })} className="min-h-10 min-w-0 flex-1 rounded-md border border-border bg-app px-2 text-content disabled:opacity-70 lg:min-h-0 lg:py-1">
                    {guideContinues && <option value="continuation">From clip #{guideContinues}</option>}
                    {formatReference && <option value="reference">From {formatReference.tag}{referenceFormatSize(formatReference) ? ` · ${referenceFormatSize(formatReference)}` : ''}</option>}
                    <option value="landscape">Landscape · 16:9</option><option value="portrait">Portrait · 9:16</option><option value="square">Square · 1:1</option>
                  </select>
                </label>
              </div>
              <VideoReferencesPanel value={reference} limits={options?.reference?.limits} history={clips} disabled={busy}
                onInsertTag={(tag) => setPrompt((p) => `${p}${p && !/\s$/.test(p) ? ' ' : ''}${tag} `)} />
            </> : <VideoSourcePicker mode={mode} onMode={setMode} frames={sources} identityReferences={useRefmods}
              aspect={aspect} onAspect={setAspect}
              onAdd={addSources} onRemove={removeSource} onClear={clearSources}
              history={clips} onOpen={openFrame}
              endFrame={endFrame} onSetEnd={setEndFrame} onClearEnd={() => setEndFrame(null)} onOpenEnd={openEndFrame} />}
            {/* ⏭ What this launch does with what is armed — in EVERY mode,
                including the two that do not run it. The strip is not even
                rendered in References mode, so a continuation staged from a
                card could otherwise go out as a fresh clip with nothing on
                screen to show it had been dropped (measured 2026-09-07). */}
            {!isReference && <div className="mt-3 space-y-2">
              {useRefmods && launchMode === 't2v' && mode === 'i2v' && <label className="flex items-center gap-2 text-xs text-content-muted">Shape
                <select aria-label="RefMods video shape" value={ASPECTS.includes(aspect) ? aspect : 'landscape'} disabled={busy}
                  onChange={e => setAspect(e.target.value)} className="rounded-md border border-border bg-app px-2 py-1 text-content">
                  <option value="landscape">16:9</option><option value="portrait">9:16</option><option value="square">1:1</option>
                </select>
              </label>}
              <VideoReferencesPanel value={reference} identitiesOnly disabled={busy}
                onInsertTag={tag => setPrompt(p => `${p}${p && !/\s$/.test(p) ? ' ' : ''}${tag} `)} />
            </div>}
            <VideoContinuationNotice state={continuation} mode={mode} isReference={isReference}
              onMode={setMode} onDrop={disarmContinuation} />
          </div>

          <div id="vs-motion" data-probe-panel="video-studio-motion" className="flex flex-col gap-1.5 rounded-xl border border-border bg-surface p-3 scroll-mt-16">
            <span className="flex flex-wrap items-center gap-1.5">
              <label htmlFor="vs-motion-text" className="text-sm font-semibold text-content">Motion</label>
              <HelpBadge topic="video-studio-motion-writer" />
              {/* ✨ Auto writes it from the start frame; ✨ Enrich rewrites what
                  is there. Both put their answer in the field and stop — the
                  render is still the user's click, and the text is still
                  theirs to edit. Auto needs a frame; without one it says so
                  rather than proposing a movement for no picture. */}
              <button type="button" onClick={autoMotion}
                disabled={busy || reference.staging || !!motionBusy || ((isReference || useRefmods) ? !reference.references.length : launchMode === 't2v' || !source.image)}
                title={(isReference || useRefmods) ? 'Write the motion using the references and their roles' : mode === 't2v'
                  ? 'Auto reads the start frame — switch to “From an image” to use it'
                  : (source.image ? 'Write the movement from the start frame'
                    : 'Pick a start frame first')}
                className="ml-auto min-h-10 rounded-lg border border-border px-2 py-1 text-[0.6875rem] text-content-muted hover:text-content disabled:opacity-40 lg:min-h-0">
                {motionBusy === 'auto' ? '…' : '✨ Auto'}
              </button>
              <button type="button" onClick={enhanceMotion}
                disabled={busy || reference.staging || !!motionBusy || !prompt.trim()}
                title="Rewrite what is written with more of the detail a sampler can use"
                className="min-h-10 rounded-lg border border-border px-2 py-1 text-[0.6875rem] text-content-muted hover:text-content disabled:opacity-40 lg:min-h-0">
                {motionBusy === 'enhance' ? '…' : '✨ Enrich'}
              </button>
              {/* ⌫ Clear: one press empties the field. A textarea that only
                  empties by select-all + delete is a chore on a phone, and the
                  two ✨ buttons read the field as a steer — a leftover sentence
                  from the last clip quietly shapes the next proposal. Disabled
                  on an empty field rather than hidden: the row keeps its shape. */}
              <button type="button" onClick={() => setPrompt('')}
                disabled={!!motionBusy || !prompt}
                title="Clear the motion field"
                aria-label="Clear the motion field"
                className="min-h-10 rounded-lg border border-border px-2 py-1 text-[0.6875rem] text-content-muted hover:text-content disabled:opacity-40 lg:min-h-0">
                ⌫ Clear
              </button>
              {/* ⚙ opens the writer window — the model and its dials belong at
                  the moment somebody wonders about them, not permanently
                  beside the buttons that use them. */}
              <button type="button" onClick={() => setModelOpen(true)}
                title="Which model writes the motion"
                aria-label="Which model writes the motion"
                className="min-h-10 rounded-lg border border-border px-2 py-1 text-[0.6875rem] text-content-muted hover:text-content lg:min-h-0">
                ⚙
              </button>
            </span>
            <textarea id="vs-motion-text" value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={5}
              placeholder={isReference ? 'The person in <Picture 1> wears the outfit from <Picture 2> and follows the movement in <Video 1>…'
                : 'What happens in the shot — she turns her head and smiles, the camera pushes in slowly…'}
              className="w-full resize-y rounded-lg border border-border bg-app px-2.5 py-2 text-sm text-content" />
            <span className="text-[0.6875rem] text-content-subtle">
              {isReference ? 'Name the tags above to say what each reference contributes. Auto and Enrich read the references together and keep those roles in H3’s reference prompt.' : <>Describe the movement, not the picture: the start frame already says
              what the scene looks like. ✨ Auto and ✨ Enrich answer in H3’s own
              three-field prompt, paced to the clip length you set.</>}
            </span>
            {isReference && reference.references.some((r) => r.kind === 'audio' || (r.kind === 'video' && r.include_audio)) && (
              <p className="text-[0.6875rem] text-amber-200">The prompt writer does not transcribe reference audio. Describe speech, rhythm or ambience in its role; the audio still conditions the generated clip.</p>
            )}
            {writerNotice && <p role="status" className="text-xs text-amber-200">{writerNotice}</p>}
            {/* 🎬 Shots. The server has always known how to cut a clip into
                timecoded shots — `shots` on every writer route — but nothing
                on this screen ever asked, so a fifteen-second clip enriched
                as one continuous take and never showed a timecode (reported
                by the maintainer, 2026-09-04). Six choices in a segmented
                row — single characters, shown all at once; a select could not
                grey the ones the length cannot hold. 1 is the default and
                today's behaviour. The row is STRETCHED like the batch's, so a
                segment stays a finger wide on a phone; the columns follow
                SHOT_CHOICES so a moved MAX_SHOTS cannot wrap the row. What the
                length cannot hold is greyed AND said in the sentence — a title
                on a disabled button never shows on a phone. */}
            <div data-testid="video-shots" className="flex flex-col gap-1 text-[0.6875rem]">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-semibold text-content">Shots</span>
                <div role="radiogroup" aria-label="Shots"
                  className="grid min-w-[16rem] flex-1 gap-1 rounded-lg border border-border bg-app p-0.5"
                  style={{ gridTemplateColumns: `repeat(${SHOT_CHOICES.length}, minmax(0, 1fr))` }}>
                  {SHOT_CHOICES.map((n) => (
                    <button key={n} type="button" role="radio" aria-checked={opts.shots === n}
                      disabled={n > shotsCap}
                      onClick={() => setOpts((o) => ({ ...o, shots: n }))}
                      title={n > shotsCap ? shotsCapNote : undefined}
                      className={`min-h-10 rounded-md px-2 py-1 text-xs font-semibold lg:min-h-0 disabled:opacity-40 ${
                        opts.shots === n ? 'bg-primary text-white' : 'text-content-muted hover:text-content'}`}>
                      {n}
                    </button>
                  ))}
                </div>
              </div>
              <span className="text-content-subtle">
                {shotsCap === 1
                  ? `Too short to cut: ${shotsCapNote}. Lengthen the clip for timecoded shots.`
                  : opts.shots > 1
                    ? `${opts.shots} shots: ✨ cuts the clip evenly, each cut written as “[Shot K] At mm:ss.mmm, the camera cuts to…”.`
                    : `One continuous take: no cuts, no timecodes. Pick 2 or more for timecoded shots${
                      shotsCap < SHOT_CHOICES.length ? ` (${shotsCapNote})` : ''}.`}
              </span>
            </div>
            {/* The presets, under the field they write into. They APPEND, like
                ✨ Enrich leaves your text alone — so the picker can be used on a
                half-written prompt without eating it. */}
            <VideoQuickPrompts mode={mode} hasReferenceImages={reference.references.some((r) => r.kind === 'image')}
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
          <VideoBestSettings state={best.state} busy={best.busy || busy} error={best.error}
            referenceMode={isReference}
            onApply={applyBest} onRemove={async () => {
              if (await best.remove()) setLora((current) => ({ ...current, runId: null, datasetId: null }));
            }} />
          <VideoOptionsPanel options={options} value={renderOpts} onChange={isReference ? reference.setSettings : setOpts} referenceMode={isReference}
            onRefresh={() => apiFetch(optionsUrl()).then(setOptions).catch((e) => toast.error(e?.message || 'Could not refresh model availability.'))} />
          <div className="flex flex-col gap-2 rounded-xl border border-border bg-surface p-3">
            <p className="break-words font-mono text-[0.6875rem] leading-snug text-content-muted">
              {readback}
            </p>
            {generateButton}
            {reason && (
              <p className="text-[0.6875rem] text-content-subtle">{reason}</p>
            )}
            {removedReference && (
              <button type="button" onClick={editRemovedReference}
                className="min-h-10 text-left text-xs text-primary underline lg:min-h-0">
                Edit missing reference in Motion
              </button>
            )}
          </div>
        </aside>
      </div>

      <section id="vs-clips" className="flex flex-col gap-2 scroll-mt-16">
        <h2 className="font-mono text-[0.625rem] uppercase tracking-[0.18em] text-content-subtle">
          Clips — newest first
        </h2>
        <AutoContinuePanel session={auto.session} ready={auto.ready} busy={auto.busy} error={auto.error}
          referenceMode={isReference}
          direction={autoDirection} maxClips={autoMaxClips} onDirection={setAutoDirection}
          onMaxClips={setAutoMaxClips} onApply={applyAuto} onStop={stopAuto} onResume={resumeAuto} />
        <VideoClipHistory clips={clips} render={renderProgress} onRate={rate} onDelete={remove} onReuse={reuse} onVfi={setVfiClip} vfiBusy={vfiBusy}
          onBest={best.save} bestBusy={best.busy} bestClipId={best.state?.best_settings?.clip_id}
          onNeuralRender={hasContributions('video.neural-render-dialog', 'studio') ? (clip) => setNrClip(clip) : null} nrBusy={nrBusy}
          onCompare={hasContributions('video.neural-compare', 'studio') ? (clip) => setCompareClip(clip) : null}
          onJumpTo={jumpTo} onContinue={continueFrom} continueBusy={continueBusy}
          onAuto={toggleAuto} autoSession={auto.session} autoMode={mode}
          autoDisabled={auto.busy || !auto.ready || (autoLimit(autoMaxClips) === null && !auto.session?.enabled)}
          autoRunning={autoIsBusy(auto.session)}
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
        <PluginSlot slot="video.neural-render-dialog" surface="studio" status={options?.neural_render} busy={nrBusy === nrClip.id}
          initial={nrClip.nr_params || null}
          subject={`Clip #${nrClip.id}${nrClip.seconds ? ` (${nrClip.seconds}s)` : ''}.`}
          consequence="The render is a NEW clip in this list; the original stays as it is."
          onRender={(params) => neuralRender(nrClip, params)}
          onClose={() => setNrClip(null)} />
      )}
      {/* ⇔ Source and render side by side, in step. The source is the row the
          render points at; if it was deleted, the left side says so. */}
      {compareClip && (
        <PluginSlot slot="video.neural-compare" surface="studio" originalSrc={clipVideoUrl(compareClip.nr_of)}
          renderSrc={clipVideoUrl(compareClip.id)}
          title={`clip #${compareClip.nr_of} → neural render #${compareClip.id}`}
          exportHref={clipComparisonUrl(compareClip.id)}
          onClose={() => setCompareClip(null)} />
      )}
      {zoom && (
        <GeneratedImageLightbox img={zoom.img} alt={zoom.index < 0 ? 'Last frame' : `Start frame ${zoom.index + 1}`}
          onClose={() => setZoom(null)}
          onImprove={canImproveCanvasImage(zoom.img) ? improveImage : undefined}
          onUseImproveSettings={restoreImproveSettings}
          datasetId={zoom.img?.dataset_id ?? null}
          onRowChanged={() => restageFrame(zoom.index, zoom.img)}
          onPrev={zoom.index > 0 ? () => openFrame(sources[zoom.index - 1], zoom.index - 1) : null}
          onNext={zoom.index >= 0 && zoom.index < sources.length - 1 ? () => openFrame(sources[zoom.index + 1], zoom.index + 1) : null} />
      )}
      {/* ⚙ The model that writes the motion, on demand. */}
      {modelOpen && (
        <MotionModelDialog onClose={() => setModelOpen(false)}
          onSaved={setMotionModel} />
      )}
    </div>
  );
}
