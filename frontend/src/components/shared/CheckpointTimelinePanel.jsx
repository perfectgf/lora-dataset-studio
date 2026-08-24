import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { apiFetch, fetchWithCsrfRetry } from '../../api/fetchClient';
import { useFocusTrap } from '../../hooks/useFocusTrap';
import {
  boundedCanvasSize, containRect, nextTimelineIndex, orderTimelineSeries,
  pickWebMMimeType, timelineFrameLabel, timelineGifError, timelineGifUrl, timelineListUrl,
  timelineLimitMessage, timelineSeriesLabel, timelineStepLabel, webmSourceLimit,
  withinWebMByteBudget, TIMELINE_PLAYBACK_MODES,
  TIMELINE_SPEEDS,
} from '../../utils/checkpointTimeline.js';

const PLAYBACK_DELAY_MS = 1400;
const CROSSFADE_MS = 500;
const EXPORT_FPS = 20;
const EXPORT_MAX_EDGE = 1280;
const EXPORT_MAX_CAPTURE_FRAMES = 600;
const EXPORT_SOURCE_CAP = 16;
const EXPORT_MAX_CHUNK_BYTES = 64 * 1024 * 1024;
const EXPORT_IMAGE_TIMEOUT_MS = 15_000;

class ExportCancelled extends Error {}

function releaseExportSource(source) {
  if (!source) return;
  if (typeof source.close === 'function') {
    source.close();
  } else if (source.tagName === 'CANVAS') {
    source.width = 0;
    source.height = 0;
  }
}

function cancelExportJob(job, error = null) {
  if (!job || job.cancelled) return;
  job.error = error || job.error || null;
  job.cancelled = true;
  for (const waiter of job.waiters) {
    clearTimeout(waiter.timer);
    waiter.reject(job.error || new ExportCancelled());
  }
  job.waiters.clear();
  for (const loader of job.loaders) loader();
  job.loaders.clear();
  for (const source of job.sources || []) releaseExportSource(source);
  job.sources?.clear();
  try {
    if (job.recorder?.state && job.recorder.state !== 'inactive') job.recorder.stop();
  } catch { /* already stopping */ }
  for (const track of job.stream?.getTracks?.() || []) track.stop();
}

function exportWait(ms, job) {
  if (job.cancelled) return Promise.reject(job.error || new ExportCancelled());
  return new Promise((resolve, reject) => {
    const waiter = { timer: null, reject };
    waiter.timer = setTimeout(() => {
      job.waiters.delete(waiter);
      if (job.cancelled) reject(job.error || new ExportCancelled());
      else resolve();
    }, ms);
    job.waiters.add(waiter);
  });
}

function preloadSameOriginImage(rawUrl, job) {
  return new Promise((resolve, reject) => {
    if (job.cancelled) { reject(new ExportCancelled()); return; }
    let url;
    try {
      url = new URL(rawUrl, window.location.href);
      if (url.origin !== window.location.origin) {
        throw new Error('A timeline frame is not served by this app, so the browser cannot export it safely.');
      }
    } catch (error) {
      reject(error instanceof Error ? error : new Error('A timeline frame has an invalid URL.'));
      return;
    }

    const image = new Image();
    const finish = (callback) => {
      clearTimeout(timeout);
      job.loaders.delete(cancel);
      image.onload = null;
      image.onerror = null;
      callback();
    };
    const cancel = () => {
      clearTimeout(timeout);
      image.onload = null;
      image.onerror = null;
      image.src = '';
      reject(new ExportCancelled());
    };
    const timeout = setTimeout(() => finish(() => {
      image.src = '';
      reject(new Error('A timeline frame took too long to load.'));
    }), EXPORT_IMAGE_TIMEOUT_MS);
    job.loaders.add(cancel);
    image.onload = () => finish(() => resolve(image));
    image.onerror = () => finish(() => reject(new Error('One of the timeline frames could not be loaded.')));
    image.decoding = 'async';
    image.src = url.href;
  });
}

function evenlySample(items, limit) {
  if (items.length <= limit) return items;
  if (limit <= 1) return [items[0]];
  return Array.from({ length: limit }, (_, index) =>
    items[Math.round(index * (items.length - 1) / (limit - 1))]);
}

function safeFilePart(value) {
  return String(value ?? 'timeline').replace(/[^a-z0-9_-]+/gi, '-').replace(/^-+|-+$/g, '') || 'timeline';
}

function triggerBrowserDownload(url, filename, dialog) {
  const focused = document.activeElement;
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.tabIndex = -1;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  if (focused?.isConnected && typeof focused.focus === 'function' && dialog?.contains(focused)) {
    focused.focus({ preventScroll: true });
  }
}

async function gifResponseError(response) {
  let serverMessage = null;
  try {
    const payload = await response.json();
    serverMessage = payload?.error || payload?.detail || payload?.message || null;
  } catch { /* the endpoint may return an empty/plain error body */ }
  const error = new Error(timelineGifError(response.status, serverMessage));
  error.status = response.status;
  return error;
}

function drawContained(context, image, width, height, opacity = 1) {
  const sourceWidth = image.naturalWidth || image.videoWidth || image.width;
  const sourceHeight = image.naturalHeight || image.videoHeight || image.height;
  const rect = containRect(sourceWidth, sourceHeight, width, height);
  context.save();
  context.globalAlpha = opacity;
  context.drawImage(image, rect.x, rect.y, rect.width, rect.height);
  context.restore();
}

async function loadBoundedExportSource(rawUrl, job) {
  const image = await preloadSameOriginImage(rawUrl, job);
  try {
    if (job.cancelled) throw new ExportCancelled();
    const size = boundedCanvasSize(image.naturalWidth, image.naturalHeight, EXPORT_MAX_EDGE);
    if (!size.width || !size.height) throw new Error('A timeline frame has no usable dimensions.');
    if (typeof globalThis.createImageBitmap === 'function') {
      try {
        const bitmap = await globalThis.createImageBitmap(image, {
          resizeWidth: size.width,
          resizeHeight: size.height,
          resizeQuality: 'high',
        });
        if (job.cancelled) {
          bitmap.close();
          throw new ExportCancelled();
        }
        return bitmap;
      } catch (error) {
        if (error instanceof ExportCancelled) throw error;
        // Older browsers may expose createImageBitmap but not resize options.
      }
    }
    const canvas = document.createElement('canvas');
    canvas.width = size.width;
    canvas.height = size.height;
    const context = canvas.getContext('2d', { alpha: false });
    if (!context) throw new Error('The browser could not prepare a timeline frame.');
    context.drawImage(image, 0, 0, size.width, size.height);
    if (job.cancelled) {
      releaseExportSource(canvas);
      throw new ExportCancelled();
    }
    return canvas;
  } finally {
    image.onload = null;
    image.onerror = null;
    image.src = '';
  }
}

function formatDate(value) {
  if (!value) return null;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  return new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(date);
}

export default function CheckpointTimelinePanel({ recordId, onClose }) {
  const [requestVersion, setRequestVersion] = useState(0);
  const [state, setState] = useState({ status: 'loading', series: [], error: null,
    limitMessage: null });
  const [selectedId, setSelectedId] = useState(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [previousIndex, setPreviousIndex] = useState(null);
  const [fadeReady, setFadeReady] = useState(true);
  const [playing, setPlaying] = useState(false);
  const [playDirection, setPlayDirection] = useState(1);
  const [playMode, setPlayMode] = useState(TIMELINE_PLAYBACK_MODES.LOOP);
  const [speed, setSpeed] = useState(1);
  const [brokenFrames, setBrokenFrames] = useState(() => new Set());
  const [gifBusy, setGifBusy] = useState(false);
  const [exportState, setExportState] = useState({ busy: false, progress: 0, message: null, error: null });
  const frameIndexRef = useRef(0);
  const fadeRafRef = useRef([]);
  const exportJobRef = useRef(null);
  const gifAbortRef = useRef(null);
  const mountedRef = useRef(true);
  const dialogRef = useRef(null);
  useFocusTrap(dialogRef, true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      gifAbortRef.current?.abort();
      cancelExportJob(exportJobRef.current);
      for (const id of fadeRafRef.current) cancelAnimationFrame(id);
    };
  }, []);

  useEffect(() => {
    const oldOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => { document.body.style.overflow = oldOverflow; };
  }, []);

  const close = useCallback(() => {
    gifAbortRef.current?.abort();
    cancelExportJob(exportJobRef.current);
    onClose?.();
  }, [onClose]);

  useEffect(() => {
    const onKeyDown = (event) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        close();
      }
    };
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [close]);

  useEffect(() => {
    let live = true;
    setState({ status: 'loading', series: [], error: null, limitMessage: null });
    setSelectedId(null);
    apiFetch(timelineListUrl(recordId))
      .then((payload) => {
        if (!live) return;
        const series = orderTimelineSeries(payload);
        setState({ status: 'ready', series, error: null,
          limitMessage: timelineLimitMessage(payload) });
        const first = series.find((item) => item.frames.length > 0) || series[0] || null;
        setSelectedId(first?.id ?? null);
      })
      .catch((error) => {
        if (!live) return;
        setState({ status: 'error', series: [],
          error: error?.message || 'Could not load this timeline.', limitMessage: null });
      });
    return () => { live = false; };
  }, [recordId, requestVersion]);

  const selectedSeries = useMemo(() => {
    if (!state.series.length) return null;
    return state.series.find((series) => String(series.id) === String(selectedId)) || state.series[0];
  }, [state.series, selectedId]);
  // useMemo: un [] neuf par rendu changeait les deps du useCallback plus bas.
  const frames = useMemo(() => selectedSeries?.frames || [], [selectedSeries]);
  const currentFrame = frames[frameIndex] || null;
  const previousFrame = previousIndex == null ? null : frames[previousIndex] || null;

  useEffect(() => {
    setFrameIndex(0);
    frameIndexRef.current = 0;
    setPreviousIndex(null);
    setFadeReady(true);
    setPlaying(false);
    setPlayDirection(1);
    setBrokenFrames(new Set());
    setExportState({ busy: false, progress: 0, message: null, error: null });
    cancelExportJob(exportJobRef.current);
    exportJobRef.current = null;
  }, [selectedSeries?.id]);

  const transitionTo = useCallback((nextIndex) => {
    const bounded = Math.max(0, Math.min(frames.length - 1, Number(nextIndex) || 0));
    const previous = frameIndexRef.current;
    if (!frames.length || bounded === previous) return;
    for (const id of fadeRafRef.current) cancelAnimationFrame(id);
    fadeRafRef.current = [];
    setPreviousIndex(previous);
    setFadeReady(false);
    frameIndexRef.current = bounded;
    setFrameIndex(bounded);
    const first = requestAnimationFrame(() => {
      const second = requestAnimationFrame(() => setFadeReady(true));
      fadeRafRef.current = [second];
    });
    fadeRafRef.current = [first];
  }, [frames.length]);

  useEffect(() => {
    if (previousIndex == null || !fadeReady) return undefined;
    const timer = setTimeout(() => setPreviousIndex(null), CROSSFADE_MS + 40);
    return () => clearTimeout(timer);
  }, [previousIndex, fadeReady, frameIndex]);

  useEffect(() => {
    if (!playing || frames.length <= 1) return undefined;
    const timer = setTimeout(() => {
      const next = nextTimelineIndex(frameIndex, frames.length, playDirection, playMode);
      setPlayDirection(next.direction);
      transitionTo(next.index);
    }, Math.max(250, PLAYBACK_DELAY_MS / speed));
    return () => clearTimeout(timer);
  }, [playing, frames.length, frameIndex, playDirection, playMode, speed, transitionTo]);

  const move = useCallback((direction) => {
    setPlaying(false);
    const next = nextTimelineIndex(frameIndex, frames.length, direction, TIMELINE_PLAYBACK_MODES.LOOP);
    setPlayDirection(direction);
    transitionTo(next.index);
  }, [frameIndex, frames.length, transitionTo]);

  const mimeType = useMemo(() => pickWebMMimeType(globalThis.MediaRecorder), []);
  const exportSupport = useMemo(() => {
    if (!mimeType || typeof document === 'undefined') return false;
    const canvas = document.createElement('canvas');
    return typeof canvas.captureStream === 'function';
  }, [mimeType]);
  const exportDisabledReason = exportSupport
    ? null
    : 'WebM export is unavailable because this browser does not support canvas video recording.';

  const exportWebM = useCallback(async () => {
    if (!exportSupport || exportState.busy || gifBusy || !selectedSeries || !frames.length) return;
    const job = { cancelled: false, error: null, waiters: new Set(), loaders: new Set(),
      sources: new Set(), recorder: null, stream: null };
    exportJobRef.current = job;
    setPlaying(false);
    setExportState({ busy: true, progress: 0, message: 'Preloading same-origin frames…', error: null });
    let objectUrl = null;

    try {
      const holdFrames = Math.max(2, Math.round(10 / speed));
      const fadeFrames = Math.max(2, Math.round(8 / speed));
      const maxSources = webmSourceLimit(
        frames.length, holdFrames, fadeFrames,
        EXPORT_MAX_CAPTURE_FRAMES, EXPORT_SOURCE_CAP,
      );
      const sourceFrames = evenlySample(frames, maxSources);
      const images = [];
      for (let index = 0; index < sourceFrames.length; index += 1) {
        if (mountedRef.current) {
          setExportState((current) => ({ ...current,
            message: `Preparing WebM source ${index + 1} of ${sourceFrames.length}…` }));
        }
        const source = await loadBoundedExportSource(sourceFrames[index].url, job);
        images.push(source);
        job.sources.add(source);
      }
      if (job.cancelled) throw new ExportCancelled();

      const firstWidth = images[0].naturalWidth || images[0].width;
      const firstHeight = images[0].naturalHeight || images[0].height;
      const size = boundedCanvasSize(firstWidth, firstHeight, EXPORT_MAX_EDGE);
      if (!size.width || !size.height) throw new Error('The first timeline frame has no usable dimensions.');
      const canvas = document.createElement('canvas');
      canvas.width = size.width;
      canvas.height = size.height;
      const context = canvas.getContext('2d', { alpha: false });
      if (!context) throw new Error('The browser could not create the export canvas.');

      const paintBackground = () => {
        context.globalAlpha = 1;
        context.fillStyle = '#080b12';
        context.fillRect(0, 0, canvas.width, canvas.height);
      };
      paintBackground();
      drawContained(context, images[0], canvas.width, canvas.height);

      const stream = canvas.captureStream(EXPORT_FPS);
      job.stream = stream;
      const recorder = new MediaRecorder(stream, {
        mimeType,
        videoBitsPerSecond: Math.min(8_000_000, Math.max(2_000_000, canvas.width * canvas.height * 5)),
      });
      job.recorder = recorder;
      const chunks = [];
      let chunkBytes = 0;
      const stopped = new Promise((resolve) => {
        recorder.ondataavailable = (event) => {
          if (!event.data?.size) return;
          if (!withinWebMByteBudget(chunkBytes, event.data.size, EXPORT_MAX_CHUNK_BYTES)) {
            cancelExportJob(job, new Error('The WebM export exceeded the 64 MiB safety limit.'));
            return;
          }
          chunkBytes += event.data.size;
          chunks.push(event.data);
        };
        recorder.onerror = () => {
          cancelExportJob(job, new Error('The browser stopped while recording the WebM export.'));
          resolve();
        };
        recorder.onstop = resolve;
      });
      recorder.start(1000);

      const totalCaptureFrames = sourceFrames.length * holdFrames
        + Math.max(0, sourceFrames.length - 1) * fadeFrames;
      let captured = 0;
      const capture = async () => {
        await exportWait(1000 / EXPORT_FPS, job);
        captured += 1;
        if (mountedRef.current) {
          setExportState((current) => ({
            ...current,
            progress: Math.min(99, Math.round(captured / totalCaptureFrames * 100)),
            message: `Rendering WebM frame ${captured} of ${totalCaptureFrames}…`,
          }));
        }
      };

      for (let index = 0; index < images.length; index += 1) {
        paintBackground();
        drawContained(context, images[index], canvas.width, canvas.height);
        for (let hold = 0; hold < holdFrames; hold += 1) await capture();
        if (index === images.length - 1) continue;
        for (let fade = 1; fade <= fadeFrames; fade += 1) {
          paintBackground();
          drawContained(context, images[index], canvas.width, canvas.height);
          drawContained(context, images[index + 1], canvas.width, canvas.height, fade / fadeFrames);
          await capture();
        }
      }

      if (job.error) throw job.error;
      if (job.cancelled) throw new ExportCancelled();
      recorder.stop();
      await stopped;
      for (const track of stream.getTracks()) track.stop();
      if (job.error) throw job.error;
      if (job.cancelled) throw new ExportCancelled();
      const blob = new Blob(chunks, { type: mimeType.split(';')[0] });
      if (!blob.size) throw new Error('The browser produced an empty WebM file.');
      objectUrl = URL.createObjectURL(blob);
      triggerBrowserDownload(objectUrl,
        `run-${safeFilePart(recordId)}-timeline-${safeFilePart(selectedSeries.id)}.webm`,
        dialogRef.current);
      await exportWait(150, job);
      URL.revokeObjectURL(objectUrl);
      objectUrl = null;

      if (mountedRef.current) {
        const boundedNote = sourceFrames.length < frames.length
          ? ` Safety limit: ${sourceFrames.length} evenly spaced source frames were used from ${frames.length}.`
          : '';
        setExportState({ busy: false, progress: 100,
          message: `WebM downloaded.${boundedNote}`, error: null });
      }
    } catch (error) {
      if (!(error instanceof ExportCancelled) && mountedRef.current) {
        setExportState({ busy: false, progress: 0, message: null,
          error: error?.message || 'Could not export this timeline.' });
      }
    } finally {
      cancelExportJob(job);
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      for (const track of job.stream?.getTracks?.() || []) track.stop();
      for (const source of job.sources) releaseExportSource(source);
      job.sources.clear();
      if (exportJobRef.current === job) exportJobRef.current = null;
    }
  }, [exportSupport, exportState.busy, gifBusy, selectedSeries, frames, speed, mimeType, recordId]);

  const downloadGif = useCallback(async () => {
    if (!selectedSeries || gifBusy || exportState.busy) return;
    const controller = new AbortController();
    gifAbortRef.current?.abort();
    gifAbortRef.current = controller;
    setGifBusy(true);
    setExportState({ busy: false, progress: 0, message: 'Rendering GIF…', error: null });
    let objectUrl = null;
    try {
      const response = await fetchWithCsrfRetry(
        timelineGifUrl(recordId, selectedSeries.id), { signal: controller.signal });
      if (!response.ok) throw await gifResponseError(response);
      const blob = await response.blob();
      if (!blob.size) throw new Error('The server produced an empty GIF file.');
      objectUrl = URL.createObjectURL(blob);
      triggerBrowserDownload(objectUrl,
        `run-${safeFilePart(recordId)}-timeline-${safeFilePart(selectedSeries.id)}.gif`,
        dialogRef.current);
      await new Promise((resolve) => setTimeout(resolve, 150));
      if (mountedRef.current) {
        setExportState({ busy: false, progress: 100, message: 'GIF downloaded.', error: null });
      }
    } catch (error) {
      if (error?.name !== 'AbortError' && mountedRef.current) {
        setExportState({ busy: false, progress: 0, message: null,
          error: error?.message || 'Could not export this timeline as a GIF.' });
      }
    } finally {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      if (gifAbortRef.current === controller) gifAbortRef.current = null;
      if (mountedRef.current) setGifBusy(false);
    }
  }, [selectedSeries, gifBusy, exportState.busy, recordId]);

  const chooseSeries = (event) => setSelectedId(event.target.value);
  const currentBroken = currentFrame && brokenFrames.has(currentFrame.id);
  const markBroken = (frame) => setBrokenFrames((current) => new Set(current).add(frame.id));
  const markLoaded = (frame) => setBrokenFrames((current) => {
    if (!current.has(frame.id)) return current;
    const next = new Set(current);
    next.delete(frame.id);
    return next;
  });
  const seriesDate = formatDate(selectedSeries?.created_at);

  return (
    <div ref={dialogRef} role="dialog" aria-modal="true" aria-labelledby="checkpoint-timeline-title"
      data-testid="checkpoint-timeline-panel"
      className="fixed inset-0 z-[90] flex items-stretch justify-center bg-black/80 p-0 backdrop-blur-sm sm:items-center sm:p-4">
      <section className="flex h-full min-h-0 w-full flex-col overflow-hidden border-border bg-surface-overlay shadow-2xl sm:max-h-[94vh] sm:max-w-5xl sm:rounded-xl sm:border">
        <header className="flex shrink-0 items-start gap-3 border-b border-border px-3 py-3 sm:px-4">
          <div className="min-w-0 flex-1">
            <h2 id="checkpoint-timeline-title" className="m-0 text-base font-semibold text-content">
              <span aria-hidden>🎞</span> Checkpoint timeline · run #{recordId}
            </h2>
            <p className="m-0 mt-1 text-[0.6875rem] text-content-muted">
              Visual crossfade only — LoRA weights are never interpolated.
            </p>
          </div>
          <button type="button" onClick={close} aria-label="Close timeline"
            className="shrink-0 rounded-md px-2 py-1 text-content-subtle hover:bg-app/50 hover:text-content">
            ✕
          </button>
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto p-3 sm:p-4">
          {state.status === 'loading' && (
            <div role="status" className="flex min-h-64 items-center justify-center text-sm text-content-muted">
              Loading checkpoint frames…
            </div>
          )}

          {state.status === 'error' && (
            <div className="mx-auto flex min-h-64 max-w-lg flex-col items-center justify-center text-center">
              <p role="alert" className="m-0 rounded-lg border border-amber-400/40 bg-amber-500/10 px-3 py-2 text-sm text-amber-100">
                {state.error}
              </p>
              <p className="m-0 mt-2 text-[0.75rem] text-content-subtle">
                The run is unchanged. Check the connection, then try loading its timeline again.
              </p>
              <button type="button" onClick={() => setRequestVersion((value) => value + 1)}
                className="mt-3 rounded-md border border-border px-3 py-2 text-sm text-content hover:border-indigo-400/60">
                Retry
              </button>
            </div>
          )}

          {state.status === 'ready' && state.series.length === 0 && (
            <div className="mx-auto flex min-h-64 max-w-lg flex-col items-center justify-center text-center">
              <p className="m-0 text-sm font-semibold text-content">No timeline frames yet.</p>
              <p className="m-0 mt-2 text-[0.75rem] text-content-muted">
                Generate previews from at least 2 checkpoints in the same launch with the same prompt,
                seed, and settings.
              </p>
            </div>
          )}

          {state.status === 'ready' && state.limitMessage && (
            <p role="status" className="mx-auto mb-3 max-w-4xl rounded-lg border border-amber-400/40 bg-amber-500/10 px-3 py-2 text-[0.6875rem] text-amber-100">
              Timeline safety limit: {state.limitMessage}
            </p>
          )}

          {state.status === 'ready' && selectedSeries && (
            <div className="mx-auto max-w-4xl">
              <div className="mb-3 flex flex-col gap-2 sm:flex-row sm:items-end">
                <label className="min-w-0 flex-1 text-[0.6875rem] font-semibold text-content-muted">
                  Preview series
                  <select value={String(selectedSeries.id)} onChange={chooseSeries}
                    aria-label="Timeline preview series"
                    className="mt-1 block w-full rounded-md border border-border bg-app px-2 py-2 text-[0.75rem] text-content">
                    {state.series.map((series, index) => (
                      <option key={series.id ?? index} value={String(series.id)}>
                        {timelineSeriesLabel(series, index)}
                      </option>
                    ))}
                  </select>
                </label>
                <div className="shrink-0 text-[0.6875rem] text-content-subtle sm:text-right">
                  <div>{selectedSeries.steps.length} checkpoint step{selectedSeries.steps.length === 1 ? '' : 's'}</div>
                  {seriesDate && <div>{seriesDate}</div>}
                </div>
              </div>

              {selectedSeries.truncated && (
                <p role="status" className="m-0 mb-3 rounded-lg border border-amber-400/40 bg-amber-500/10 px-3 py-2 text-[0.6875rem] text-amber-100">
                  Showing {frames.length} of {selectedSeries.frame_count} frames. The server capped this series;
                  the player and exports use only the frames listed here.
                </p>
              )}

              {frames.length === 0 ? (
                <div className="flex min-h-64 items-center justify-center rounded-xl border border-border bg-app/40 px-4 text-center text-sm text-content-muted">
                  This preview series has no available frames. Choose another series, or generate previews
                  from at least 2 checkpoints in the same launch with the same prompt, seed, and settings.
                </div>
              ) : (
                <>
                  <div className="relative flex min-h-[18rem] items-center justify-center overflow-hidden rounded-xl border border-border bg-[#080b12] sm:min-h-[26rem]"
                    aria-live="polite" aria-label={timelineFrameLabel(currentFrame, frameIndex, frames.length)}>
                    {previousFrame && (
                      <img src={previousFrame.url} alt="" aria-hidden
                        className={`pointer-events-none absolute inset-0 h-full w-full object-contain transition-opacity duration-500 ease-linear motion-reduce:transition-none ${fadeReady ? 'opacity-0' : 'opacity-100'}`} />
                    )}
                    {currentFrame && (
                      <img key={`${selectedSeries.id}:${currentFrame.id}`} src={currentFrame.url}
                        alt={timelineFrameLabel(currentFrame, frameIndex, frames.length)}
                        onLoad={() => markLoaded(currentFrame)}
                        onError={() => markBroken(currentFrame)}
                        className={`absolute inset-0 h-full w-full object-contain transition-opacity duration-500 ease-linear motion-reduce:transition-none ${previousFrame && !fadeReady ? 'opacity-0' : 'opacity-100'}`} />
                    )}
                    {currentBroken && (
                      <p role="alert" className="relative z-10 rounded-lg border border-amber-400/40 bg-black/80 px-3 py-2 text-sm text-amber-100">
                        This frame could not be loaded. Use the controls to continue to another checkpoint.
                      </p>
                    )}
                    <div className="pointer-events-none absolute inset-x-0 bottom-0 flex items-end justify-between bg-gradient-to-t from-black/80 to-transparent px-3 pb-2 pt-8 text-white">
                      <span className="text-sm font-semibold tabular-nums">{timelineStepLabel(currentFrame?.step)}</span>
                      <span className="text-[0.75rem] tabular-nums">{frameIndex + 1} / {frames.length}</span>
                    </div>
                  </div>

                  <label className="mt-3 block text-[0.6875rem] font-semibold text-content-muted">
                    Scrub timeline
                    <input type="range" min="0" max={Math.max(0, frames.length - 1)} value={frameIndex}
                      onChange={(event) => { setPlaying(false); transitionTo(Number(event.target.value)); }}
                      aria-valuetext={timelineFrameLabel(currentFrame, frameIndex, frames.length)}
                      className="mt-1 block w-full accent-indigo-500" />
                  </label>

                  <div aria-label="Timeline frames" className="mt-2 flex gap-2 overflow-x-auto pb-2">
                    {frames.map((frame, index) => (
                      <button key={frame.id ?? index} type="button" onClick={() => { setPlaying(false); transitionTo(index); }}
                        aria-current={index === frameIndex ? 'true' : undefined}
                        aria-label={`Go to ${timelineFrameLabel(frame, index, frames.length)}`}
                        title={timelineFrameLabel(frame, index, frames.length)}
                        className={`relative h-16 w-16 shrink-0 overflow-hidden rounded-md border bg-[#080b12] ${index === frameIndex
                          ? 'border-indigo-300 ring-2 ring-indigo-400/60'
                          : 'border-border hover:border-indigo-400/60'}`}>
                        <img src={frame.url} alt="" loading="lazy" className="h-full w-full object-contain" />
                        <span aria-hidden className="absolute inset-x-0 bottom-0 bg-black/75 px-1 py-0.5 text-[0.5625rem] text-white tabular-nums">
                          {frame.step == null ? '?' : Number(frame.step).toLocaleString('en-US')}
                        </span>
                      </button>
                    ))}
                  </div>

                  <div className="mt-2 flex flex-wrap items-center gap-2 border-t border-border pt-3">
                    <div className="flex items-center gap-1" aria-label="Playback controls">
                      <button type="button" onClick={() => move(-1)} aria-label="Previous frame"
                        className="rounded-md border border-border px-3 py-2 text-content hover:border-indigo-400/60">◀</button>
                      <button type="button" onClick={() => setPlaying((value) => !value)}
                        aria-label={playing ? 'Pause timeline' : 'Play timeline'} aria-pressed={playing}
                        className="min-w-20 rounded-md border border-indigo-400/60 bg-indigo-500/15 px-3 py-2 text-[0.75rem] font-semibold text-indigo-100 hover:bg-indigo-500/25">
                        {playing ? '❚❚ Pause' : '▶ Play'}
                      </button>
                      <button type="button" onClick={() => move(1)} aria-label="Next frame"
                        className="rounded-md border border-border px-3 py-2 text-content hover:border-indigo-400/60">▶</button>
                    </div>

                    <button type="button"
                      onClick={() => { setPlayMode((mode) => mode === TIMELINE_PLAYBACK_MODES.LOOP
                        ? TIMELINE_PLAYBACK_MODES.PING_PONG : TIMELINE_PLAYBACK_MODES.LOOP); setPlayDirection(1); }}
                      aria-label={`Playback mode: ${playMode === TIMELINE_PLAYBACK_MODES.LOOP ? 'loop' : 'ping-pong'}`}
                      title="Toggle loop or ping-pong playback"
                      className="rounded-md border border-border px-2.5 py-2 text-[0.6875rem] text-content-muted hover:border-indigo-400/60 hover:text-content">
                      {playMode === TIMELINE_PLAYBACK_MODES.LOOP ? '↻ Loop' : '↔ Ping-pong'}
                    </button>

                    <div role="group" aria-label="Playback speed" className="flex items-center rounded-md border border-border p-0.5">
                      {TIMELINE_SPEEDS.map((rate) => (
                        <button key={rate} type="button" onClick={() => setSpeed(rate)} aria-pressed={speed === rate}
                          className={`rounded px-2 py-1.5 text-[0.6875rem] ${speed === rate
                            ? 'bg-indigo-500/30 font-semibold text-indigo-100'
                            : 'text-content-subtle hover:text-content'}`}>
                          {rate}×
                        </button>
                      ))}
                    </div>

                    <div className="ml-auto flex flex-wrap items-center justify-end gap-2">
                      <button type="button" onClick={downloadGif}
                        aria-disabled={gifBusy || exportState.busy}
                        className="rounded-md border border-border px-3 py-2 text-[0.75rem] text-content-muted hover:border-indigo-400/60 hover:text-content aria-disabled:cursor-not-allowed aria-disabled:opacity-40"
                        aria-label="Download timeline as GIF">
                        {gifBusy ? 'Rendering GIF…' : 'Download GIF'}
                      </button>
                      <button type="button" onClick={exportWebM}
                        disabled={!exportSupport}
                        aria-disabled={!exportSupport || exportState.busy || gifBusy}
                        title={exportDisabledReason || 'Render this timeline locally and download a WebM video'}
                        className="rounded-md border border-border px-3 py-2 text-[0.75rem] text-content-muted hover:border-indigo-400/60 hover:text-content disabled:cursor-not-allowed disabled:opacity-40 aria-disabled:cursor-not-allowed aria-disabled:opacity-40">
                        {exportState.busy ? `Exporting ${exportState.progress}%…` : 'Export WebM'}
                      </button>
                    </div>
                  </div>

                  {exportDisabledReason && (
                    <p className="m-0 mt-2 text-right text-[0.625rem] text-content-subtle">
                      {exportDisabledReason}
                    </p>
                  )}
                  {(exportState.busy || exportState.message) && (
                    <p role="status" aria-live="polite" className="m-0 mt-2 text-right text-[0.6875rem] text-content-muted">
                      {exportState.message}
                    </p>
                  )}
                  {exportState.error && (
                    <p role="alert" className="m-0 mt-2 rounded-md border border-rose-400/40 bg-rose-500/10 px-2 py-1.5 text-right text-[0.6875rem] text-rose-100">
                      {exportState.error}
                    </p>
                  )}
                </>
              )}
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
