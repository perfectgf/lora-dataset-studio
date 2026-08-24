/* 🗑 "Delete this run and everything it produced" — the words and the numbers.
 *
 * The gesture lives on the LoRA Canvas, a board where cards get dragged around
 * all day. So the confirmation cannot be a generic "Are you sure?": it has to
 * COUNT what is about to disappear ("14 checkpoints · 22.4 GB, 37 images") and
 * name what survives, loudly enough that a user can back out by reading.
 *
 * Every decision worth getting wrong is here rather than in the panel, because
 * `node --test` does not parse JSX and these are exactly the sentences that must
 * not be eyeballed: what is deletable at all, what the dialog promises, and how
 * a PARTIAL result is reported (a half-done destruction shown as a success is
 * the worst outcome this feature can have).
 */

/** Bytes as a short human string. Binary units, because that is what a file
 *  manager shows for a .safetensors — "22.4 GB" beside a 24-billion-byte file
 *  would read as a different file. '' for nothing/unknown, so callers can drop
 *  the clause entirely instead of printing "0 B". */
export function formatBytes(n) {
  const v = Number(n);
  if (!Number.isFinite(v) || v <= 0) return '';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  let x = v;
  while (x >= 1024 && i < units.length - 1) { x /= 1024; i += 1; }
  const digits = i === 0 || x >= 100 ? 0 : 1;
  return `${x.toFixed(digits)} ${units[i]}`;
}

function plural(n, one, many) {
  return `${n} ${n === 1 ? one : many}`;
}

function count(impact, key) {
  const v = Number(impact?.[key]);
  return Number.isFinite(v) && v > 0 ? v : 0;
}

/* The cascade half of the deletion-impact payload. Missing (an older backend, a
   failed probe) yields an all-zero block rather than undefined, so the caller
   never has to guard every read. */
function cascadeBlock(impact) {
  return impact?.cascade || {};
}

/** Can this run be cascade-deleted right now, and if not, why not?
 *
 *  A run that is TRAINING is refused here as well as in the backend: the button
 *  says why instead of offering a click that can only 409. Anything else is
 *  allowed — unlike the conservative "remove a gone run", the whole point is
 *  that a run WITH checkpoints on disk can be deleted. */
export function cascadeBlockedReason(impact) {
  const active = cascadeBlock(impact).training_active;
  if (active === 'cloud') return 'A cloud pod is still training this run.';
  if (active) return 'This dataset is training right now.';
  return null;
}

/** Lines listing what the cascade DESTROYS, most alarming first. Zero counts are
 *  omitted — "0 images" is noise that buries the line that matters. */
export function cascadeLosses(impact) {
  const c = cascadeBlock(impact);
  const lines = [];
  const cks = count(c, 'checkpoints');
  if (cks) {
    const size = formatBytes(c.checkpoint_bytes);
    lines.push(`${plural(cks, 'checkpoint', 'checkpoints')}${size ? ` · ${size}` : ''}`);
  }
  const imgs = count(c, 'images_deleted');
  if (imgs) lines.push(`${plural(imgs, 'generated image', 'generated images')}`);
  const notes = count(impact, 'notes');
  if (notes) lines.push(plural(notes, 'checkpoint note', 'checkpoint notes'));
  const previews = count(impact, 'previews');
  if (previews) lines.push(plural(previews, 'preview link', 'preview links'));
  if (count(impact, 'canvas_positions')) lines.push('its place on the canvas');
  const arch = count(impact, 'archived_images_released');
  if (arch) {
    lines.push(`${plural(arch, 'archived source image', 'archived source images')}`
      + ' (kept only for this run)');
  }
  return lines;
}

/** What SURVIVES, stated positively. These are the four things a user would be
 *  angry to lose silently, so each one is spelled out rather than implied. */
export function cascadeKeeps(impact) {
  const c = cascadeBlock(impact);
  const lines = [];
  const kids = count(impact, 'children_detached');
  if (kids) {
    lines.push(kids === 1
      ? '1 run that continued from it is kept, as its own root.'
      : `${kids} runs that continued from it are kept, as their own roots.`);
  }
  const kept = count(c, 'images_kept_rated');
  if (kept) {
    lines.push(kept === 1
      ? '1 image you marked 👍 is kept — it only loses the link to this run.'
      : `${kept} images you marked 👍 are kept — they only lose the link to this run.`);
  }
  const dep = count(c, 'deployed_kept');
  if (dep) {
    lines.push(`${plural(dep, 'LoRA', 'LoRAs')} already deployed into ComfyUI `
      + `${dep === 1 ? 'stays' : 'stay'} there — undeploy from Checkpoints & LoRAs `
      + `if you want ${dep === 1 ? 'it' : 'them'} gone.`);
  }
  if (!lines.length) lines.push('Nothing else in the app points at this run.');
  return lines;
}

/** Everything the confirmation dialog renders, decided in one place. */
export function cascadeConfirmation(recordId, impact) {
  return {
    title: `Delete run #${recordId} and everything it produced?`,
    losses: cascadeLosses(impact),
    keeps: cascadeKeeps(impact),
    // An impact we could not read is stated as such: the delete is still
    // offered (the backend is the real guard) but with no invented numbers.
    unknown: !impact,
    blockedReason: cascadeBlockedReason(impact),
  };
}

/** One-line summary of what the API actually did.
 *
 *  A 409 'partial' never reaches here — it throws — but a 200 can still carry a
 *  smaller number than the dialog promised (an image already gone, a row still
 *  generating). Saying "Run deleted" over that would hide it, so the counts come
 *  back with the message. */
export function cascadeResultMessage(res) {
  const cks = count(res, 'checkpoints_deleted');
  const imgs = count(res, 'images_deleted');
  const kept = count(res, 'images_kept');
  const bits = [];
  if (cks) bits.push(plural(cks, 'checkpoint', 'checkpoints'));
  if (imgs) bits.push(plural(imgs, 'image', 'images'));
  let msg = bits.length ? `Run deleted — ${bits.join(' and ')} removed.` : 'Run deleted.';
  if (kept) msg += ` ${plural(kept, 'image', 'images')} kept.`;
  return msg;
}
