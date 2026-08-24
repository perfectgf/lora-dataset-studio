/* Pure decisions for the checkpoint timeline.
 *
 * Keeping URLs, ordering and playback edges outside the React component makes
 * the two easy-to-miss promises testable: a frame at the end really loops (or
 * reverses), and an export can only ever be WebM. */

export const TIMELINE_PLAYBACK_MODES = Object.freeze({
  LOOP: 'loop',
  PING_PONG: 'ping-pong',
});

export const TIMELINE_SPEEDS = Object.freeze([0.5, 1, 2]);

const WEBM_MIME_CANDIDATES = Object.freeze([
  'video/webm;codecs=vp9',
  'video/webm;codecs=vp8',
  'video/webm',
]);

const finiteNumber = (value) => {
  if (value == null || value === '') return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
};

const dateValue = (value) => {
  const parsed = value == null ? Number.NaN : Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
};

const SQLITE_INT64_MAX = 9223372036854775807n;

const safeRecordId = (value) => {
  if (typeof value === 'number' && !Number.isSafeInteger(value)) return null;
  const text = String(value ?? '').trim();
  if (!/^[1-9][0-9]*$/.test(text)) return null;
  try {
    return BigInt(text) <= SQLITE_INT64_MAX ? text : null;
  } catch {
    return null;
  }
};

export function timelineListUrl(recordId) {
  const safeId = safeRecordId(recordId);
  return safeId ? `/api/train/run/${safeId}/timeline` : null;
}

export function timelineGifUrl(recordId, seriesId) {
  const list = timelineListUrl(recordId);
  const safeSeriesId = String(seriesId ?? '');
  if (!list || !/^[0-9a-f]{64}$/.test(safeSeriesId)) return null;
  return `${list}/${safeSeriesId}/gif`;
}

export function timelineGifError(status, serverMessage = null) {
  if (typeof serverMessage === 'string' && serverMessage.trim()) return serverMessage.trim();
  if (status === 429) return 'The GIF renderer is busy. Wait a moment and try again.';
  if (status === 413) return 'This timeline is too large to export as a GIF.';
  if (status === 404) return 'This timeline is no longer available. Refresh it and try again.';
  return `Could not export this timeline as a GIF${status ? ` (HTTP ${status})` : ''}.`;
}

export function timelineEndpoints(recordId, seriesId = null) {
  const list = timelineListUrl(recordId);
  if (!list) return null;
  return { list, gif: timelineGifUrl(recordId, seriesId) };
}

export function orderTimelineFrames(frames) {
  if (!Array.isArray(frames)) return [];
  return frames
    .map((frame, sourceIndex) => ({ ...frame, __sourceIndex: sourceIndex }))
    .sort((left, right) => {
      const leftStep = finiteNumber(left.step);
      const rightStep = finiteNumber(right.step);
      if (leftStep != null || rightStep != null) {
        if (leftStep == null) return 1;
        if (rightStep == null) return -1;
        if (leftStep !== rightStep) return leftStep - rightStep;
      }

      const leftDate = dateValue(left.created_at);
      const rightDate = dateValue(right.created_at);
      if (leftDate != null || rightDate != null) {
        if (leftDate == null) return 1;
        if (rightDate == null) return -1;
        if (leftDate !== rightDate) return leftDate - rightDate;
      }

      const leftId = finiteNumber(left.id);
      const rightId = finiteNumber(right.id);
      if (leftId != null && rightId != null && leftId !== rightId) return leftId - rightId;
      return left.__sourceIndex - right.__sourceIndex;
    })
    .map(({ __sourceIndex, ...frame }) => frame);
}

function derivedCreatedAt(frames) {
  return frames.find((frame) => dateValue(frame.created_at) != null)?.created_at || null;
}

function derivedSteps(frames) {
  return [...new Set(frames
    .map((frame) => finiteNumber(frame.step))
    .filter((step) => step != null))];
}

export function orderTimelineSeries(payload) {
  const input = Array.isArray(payload)
    ? payload
    : (Array.isArray(payload?.series) ? payload.series : []);

  return input
    .map((series, sourceIndex) => {
      const frames = orderTimelineFrames(series?.frames);
      const suppliedSteps = Array.isArray(series?.steps)
        ? series.steps.map(finiteNumber).filter((step) => step != null)
        : [];
      const steps = [...new Set(suppliedSteps.length ? suppliedSteps : derivedSteps(frames))]
        .sort((a, b) => a - b);
      return {
        ...(series || {}),
        frames,
        steps,
        created_at: series?.created_at || derivedCreatedAt(frames),
        frame_count: Math.max(finiteNumber(series?.frame_count) ?? frames.length, frames.length),
        shown: finiteNumber(series?.shown) ?? frames.length,
        truncated: Boolean(series?.truncated),
        __sourceIndex: sourceIndex,
      };
    })
    .sort((left, right) => {
      const leftDate = dateValue(left.created_at);
      const rightDate = dateValue(right.created_at);
      if (leftDate != null || rightDate != null) {
        if (leftDate == null) return 1;
        if (rightDate == null) return -1;
        if (leftDate !== rightDate) return rightDate - leftDate;
      }
      const leftId = finiteNumber(left.id);
      const rightId = finiteNumber(right.id);
      if (leftId != null && rightId != null && leftId !== rightId) return rightId - leftId;
      return left.__sourceIndex - right.__sourceIndex;
    })
    .map(({ __sourceIndex, ...series }) => series);
}

export function timelineLimitMessage(payload) {
  if (!payload || payload.truncated !== true) return null;
  const parts = [];
  const count = Math.max(0, finiteNumber(payload.count) ?? 0);
  const shown = Math.max(0, finiteNumber(payload.shown) ?? 0);
  if (count > shown) parts.push(`${shown} of ${count} preview series shown.`);

  const candidates = Math.max(0, finiteNumber(payload.candidate_count) ?? 0);
  const scanned = Math.max(0, finiteNumber(payload.candidates_scanned) ?? candidates);
  if (candidates > scanned) parts.push(`${scanned} of ${candidates} candidate frames scanned.`);

  const frameCount = Math.max(0, finiteNumber(payload.frame_count) ?? 0);
  const framesShown = Math.max(0, finiteNumber(payload.frames_shown) ?? frameCount);
  if (frameCount > framesShown) parts.push(`${framesShown} of ${frameCount} comparable frames shown.`);
  if (!parts.length) parts.push('Some timeline results were omitted by server safety limits.');
  return parts.join(' ');
}

export function nextTimelineIndex(index, count, direction = 1,
  mode = TIMELINE_PLAYBACK_MODES.LOOP) {
  const size = Math.max(0, Math.floor(Number(count) || 0));
  if (size <= 1) return { index: 0, direction: 1 };

  const current = Math.min(size - 1, Math.max(0, Math.floor(Number(index) || 0)));
  const travel = Number(direction) < 0 ? -1 : 1;
  const candidate = current + travel;

  if (candidate >= 0 && candidate < size) {
    return { index: candidate, direction: travel };
  }
  if (mode === TIMELINE_PLAYBACK_MODES.PING_PONG) {
    const reversed = travel * -1;
    return { index: current + reversed, direction: reversed };
  }
  return { index: candidate < 0 ? size - 1 : 0, direction: travel };
}

export function timelineStepLabel(step) {
  const number = finiteNumber(step);
  return number == null ? 'Step unknown' : `Step ${number.toLocaleString('en-US')}`;
}

export function timelineFrameLabel(frame, index, count) {
  const size = Math.max(0, Math.floor(Number(count) || 0));
  const position = size ? Math.min(size, Math.max(1, Math.floor(Number(index) || 0) + 1)) : 0;
  return `${timelineStepLabel(frame?.step)} · ${position} of ${size}`;
}

function compactConditionParts(conditions) {
  if (!conditions || typeof conditions !== 'object' || Array.isArray(conditions)) return [];
  const parts = [];
  if (conditions.seed != null && conditions.seed !== '') parts.push(`Seed ${conditions.seed}`);
  if (conditions.strength != null && conditions.strength !== '') {
    parts.push(`strength ${conditions.strength}`);
  }
  if (conditions.aspect != null && String(conditions.aspect).trim()) {
    parts.push(String(conditions.aspect).trim());
  }
  return parts;
}

export function timelineSeriesLabel(series, index = 0) {
  const parts = [`Series ${Math.max(0, Number(index) || 0) + 1}`,
    ...compactConditionParts(series?.conditions)];
  const count = Math.max(0, finiteNumber(series?.frame_count) ?? series?.frames?.length ?? 0);
  parts.push(`${count} frame${count === 1 ? '' : 's'}`);
  return parts.join(' · ');
}

export function pickWebMMimeType(MediaRecorderClass = globalThis.MediaRecorder) {
  if (typeof MediaRecorderClass !== 'function') return null;
  if (typeof MediaRecorderClass.isTypeSupported !== 'function') return 'video/webm';
  return WEBM_MIME_CANDIDATES.find((mime) => MediaRecorderClass.isTypeSupported(mime)) || null;
}

export function containRect(sourceWidth, sourceHeight, targetWidth, targetHeight) {
  const sw = finiteNumber(sourceWidth);
  const sh = finiteNumber(sourceHeight);
  const tw = finiteNumber(targetWidth);
  const th = finiteNumber(targetHeight);
  if (!(sw > 0) || !(sh > 0) || !(tw > 0) || !(th > 0)) {
    return { x: 0, y: 0, width: 0, height: 0 };
  }
  const scale = Math.min(tw / sw, th / sh);
  const width = sw * scale;
  const height = sh * scale;
  return { x: (tw - width) / 2, y: (th - height) / 2, width, height };
}

export function boundedCanvasSize(width, height, maxEdge = 1280) {
  const sourceWidth = finiteNumber(width);
  const sourceHeight = finiteNumber(height);
  const edge = Math.max(1, finiteNumber(maxEdge) ?? 1280);
  if (!(sourceWidth > 0) || !(sourceHeight > 0)) return { width: 0, height: 0 };
  const scale = Math.min(1, edge / Math.max(sourceWidth, sourceHeight));
  return {
    width: Math.max(1, Math.round(sourceWidth * scale)),
    height: Math.max(1, Math.round(sourceHeight * scale)),
  };
}

export function webmSourceLimit(frameCount, holdFrames, fadeFrames,
  captureFrameCap = 600, sourceCap = 16) {
  const available = Math.max(0, Math.floor(finiteNumber(frameCount) ?? 0));
  if (!available) return 0;
  const hold = Math.max(1, Math.floor(finiteNumber(holdFrames) ?? 1));
  const fade = Math.max(0, Math.floor(finiteNumber(fadeFrames) ?? 0));
  const captureCap = Math.max(1, Math.floor(finiteNumber(captureFrameCap) ?? 1));
  const decodedCap = Math.max(1, Math.floor(finiteNumber(sourceCap) ?? 1));
  // n * hold + (n - 1) * fade <= captureCap
  const captureLimited = Math.max(1, Math.floor((captureCap + fade) / (hold + fade)));
  return Math.min(available, decodedCap, captureLimited);
}

export function withinWebMByteBudget(currentBytes, nextBytes,
  byteCap = 64 * 1024 * 1024) {
  const current = finiteNumber(currentBytes);
  const next = finiteNumber(nextBytes);
  const cap = finiteNumber(byteCap);
  return current != null && next != null && cap != null
    && current >= 0 && next >= 0 && cap >= 0 && current + next <= cap;
}
