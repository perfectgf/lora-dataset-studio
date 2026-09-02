/* The start frames of a launch, and the clips queued from them — the logic,
 * kept out of the components so `node --test` can exercise it.
 *
 * One frame was the whole story until 2026-09-02: a LoRA judged on one
 * portrait is a LoRA judged on one portrait, and the ask was to queue a clip
 * for EACH of several pictures in one click. The launch stays one clip per
 * POST — the server renders one clip per graph, and that is the contract the
 * history, the poller and ↻ Reuse all read — what changes is that the panel
 * walks a list, on one seed, and stops at the first refusal.
 */
import { buildGeneratePayload } from './videoStudioApi.js';

/** A frame is `{ key, image, ratio, preview }`: the staged NAME the graph
 * loads, its aspect, a picture for the strip — and a key that says where it
 * CAME from (`bank:3:41`, `gallery:88`, `clip:2:walk.mp4`, an upload's name,
 * size and date). Never the staged name: the server stages every pick under a
 * fresh uuid, so the same tile pressed twice would be two files and two
 * identical clips. The key is the ORIGIN, not the picture: the same portrait
 * reached by two routes — its Bank tile and its Gallery tile, or a ↻ Reuse
 * frame (`staged:…`) and the tile it came from — is two keys, and two clips
 * on one seed; the pressed tiles show what the strip holds, and hashing every
 * pick to catch that case would read each file twice. Appended in pick order;
 * what the list already holds is skipped, and the caller can say so. */
export function addFrames(list, incoming) {
  const out = [...(list || [])];
  const seen = new Set(out.map((f) => f.key));
  const skipped = [];
  for (const frame of incoming || []) {
    if (!frame || !frame.image) continue;
    if (seen.has(frame.key)) { skipped.push(frame); continue; }
    seen.add(frame.key);
    out.push(frame);
  }
  return { frames: out, skipped };
}

export function removeFrame(list, key) {
  return (list || []).filter((f) => f.key !== key);
}

/** The key of an upload: the file, not its bytes — reading a file to hash it
 * before the upload that reads it again is a cost for a guard that only has to
 * catch the same file chosen twice from the same dialog. */
export function uploadKey(file) {
  return `upload:${file.name}:${file.size}:${file.lastModified}`;
}

/** Let go of a frame's picture when it is a blob URL the picker minted for an
 * upload: the browser keeps the File alive for as long as the URL is
 * registered, and a session of picks would otherwise hold every file it ever
 * chose — the ones the strip dropped and the ones the dedupe refused included.
 * A server URL is not ours to revoke. */
export function releasePreview(frame) {
  const url = frame?.preview;
  if (typeof url === 'string' && url.startsWith('blob:')
      && typeof URL !== 'undefined' && typeof URL.revokeObjectURL === 'function') {
    URL.revokeObjectURL(url);
  }
}

/** What the batch says about enrichment when the first reply names no prompt:
 * the rest ran the prompt as typed, and the notice has to say so — the
 * alternative is a batch that claims one prompt over clips that ran two. */
export const ENRICH_UNKNOWN = 'the server did not say which prompt ran, so the later clips ran the prompt as typed';

/** Queue one clip per frame, in order, on ONE seed and ONE prompt.
 *
 * `base` is the launch without its frame (mode, prompt, LoRA, dials — and the
 * seed as typed, '' for random); `post(body)` performs one POST /generate and
 * resolves to the server's reply; `onQueued(done, total)` is called after
 * each launch that went through, for a button that counts.
 *
 * What the SERVER ran for the first clip is what the rest run: its seed, and
 * its prompt. The seed, because a dial left on random — or on a negative
 * number, which the server draws on too — must draw once, not once per frame,
 * so the clips differ by their frame and by nothing else, the rule ↻ Reuse
 * lives by; the reply's `seed` is what the route wrote into the graph, so it
 * is adopted whatever was typed. The prompt, because ✨ Enrich at launch runs
 * in a vision window that a clip already queued shuts (refuted 2026-09-02:
 * asking on every launch gave the first clip the rewrite and the rest the
 * prompt as typed, under one toast that named neither), so the rewrite is
 * asked for once and the reply's `prompt` travels with `enhance` dropped. A
 * reply without a seed leaves the later launches random; one without a
 * prompt leaves them as typed; in both cases the notice says so rather than
 * keeping a promise it cannot check. A frame of `null` is a launch without a
 * picture (text-only mode passes one).
 *
 * Fail-fast: the first refusal ends the walk. What queued stays queued — the
 * server has them — and the outcome says how far it got, so the notice can
 * say "Queued 2 of 5" rather than "failed" over three clips that are
 * rendering. */
export async function queueClips(frames, base, post, onQueued = () => {}) {
  const launches = (frames && frames.length) ? frames : [null];
  const queued = [];
  let seed = base?.seed;
  let prompt = base?.prompt;
  let enhance = !!base?.enhance;
  let enrichSkipped = null;
  for (const frame of launches) {
    let reply;
    try {
      reply = await post(buildGeneratePayload({
        ...base, seed, prompt, enhance, image: frame?.image || null, ratio: frame?.ratio || null,
      }));
    } catch (error) {
      return { queued, total: launches.length, seed, failed: error, enrichSkipped };
    }
    queued.push(reply || {});
    if (reply?.seed !== undefined && reply?.seed !== null) seed = reply.seed;
    // The launch went through with the prompt as typed — the writer could not
    // run (fence, server away). Kept once: the same reason five times is noise.
    if (reply?.enrich_skipped && !enrichSkipped) enrichSkipped = reply.enrich_skipped;
    if (enhance) {
      enhance = false;
      if (typeof reply?.prompt === 'string' && reply.prompt.trim()) {
        prompt = reply.prompt;
      } else if (!reply?.enrich_skipped && launches.length > 1 && !enrichSkipped) {
        enrichSkipped = ENRICH_UNKNOWN;
      }
    }
    onQueued(queued.length, launches.length);
  }
  return { queued, total: launches.length, seed, failed: null, enrichSkipped };
}

/** What the toast says when every launch went through. One clip reads as it
 * always did ("Queued — seed 7, 56 frames."); a batch counts itself, and a
 * batch whose seed could not be shared — the first reply named none — says
 * so, since "on one seed" is written on the strip and in the Guide. */
export function queuedNotice(outcome) {
  const n = outcome.queued.length;
  const head = n > 1 ? `Queued ${n} clips` : 'Queued';
  const bits = [];
  if (outcome.seed !== '' && outcome.seed !== null && outcome.seed !== undefined) {
    bits.push(`seed ${outcome.seed}`);
  } else if (n > 1) {
    bits.push('independent seeds (the server did not return the first)');
  }
  const frames = outcome.queued[0]?.frames;
  if (frames) bits.push(`${frames} frames`);
  return bits.length ? `${head} — ${bits.join(', ')}.` : `${head}.`;
}

/** What the toast says when a launch was refused: the count first, when
 * some went through, then the server's own sentence. */
export function failureNotice(outcome, fallback = 'The clip could not be queued.') {
  const message = outcome.failed?.message || fallback;
  const k = outcome.queued.length;
  return k > 0 ? `Queued ${k} of ${outcome.total} — ${message}` : message;
}

/** The button's text: how many clips this click queues, and while it queues,
 * how far it is. Text-only is always one clip whatever the strip holds. */
export function generateLabel({ mode, count, busy, done = 0, total = 0 }) {
  if (busy) {
    return total > 1 ? `Queueing ${Math.min(done + 1, total)} of ${total}…` : 'Queueing…';
  }
  const n = mode === 't2v' ? 1 : count;
  return n > 1 ? `Generate ${n} clips` : 'Generate clip';
}
