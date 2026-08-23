// Guest checkpoints in the 1-LoRA Test Studio: files from models/loras that
// are not this dataset's trigger-matched pool, compared as their own cells
// (same prompt and seed) rather than stacked extras.
//
// Shape: [{filename, label}]. Cap matches the Canvas plugin-node limit so a
// picker dump cannot explode the matrix. JSX-free for node --test.

export const MAX_GUEST_CHECKPOINTS = 16;
export const GUEST_LABEL_PREFIX = 'Theirs · ';

export function guestStem(filename) {
  const parts = String(filename || '').replace(/\\/g, '/').split('/');
  return (parts[parts.length - 1] || String(filename || '')).replace(/\.safetensors$/i, '');
}

export function guestLabel(filename) {
  return GUEST_LABEL_PREFIX + guestStem(filename);
}

export function normalizeGuestCheckpoints(raw) {
  if (!Array.isArray(raw)) return [];
  const out = [];
  const seen = new Set();
  for (const e of raw) {
    const filename = typeof e === 'string' ? e.trim()
      : String(e?.filename || '').trim();
    if (!filename || seen.has(filename)) continue;
    seen.add(filename);
    out.push({ filename, label: guestLabel(filename) });
    if (out.length >= MAX_GUEST_CHECKPOINTS) break;
  }
  return out;
}

/** Mine ticks + guest ticks, or the canvas's pinned list unchanged. */
export function chosenCheckpoints({ mineFns, selCps, guests, selGuests, pinned }) {
  if (pinned) return pinned;
  const mine = mineFns || [];
  const mineChosen = (selCps ?? mine).filter((fn) => mine.includes(fn));
  const guestFns = (guests || []).map((g) => g.filename);
  const guestChosen = (selGuests ?? guestFns).filter(
    (fn) => guestFns.includes(fn) && !mine.includes(fn));
  return [...mineChosen, ...guestChosen];
}

export function addGuestCheckpoint(guests, filename, mineFns = []) {
  const fn = String(filename || '').trim();
  if (!fn) return guests || [];
  if ((mineFns || []).includes(fn)) return guests || [];
  const cur = guests || [];
  if (cur.some((g) => g.filename === fn)) return cur;
  if (cur.length >= MAX_GUEST_CHECKPOINTS) return cur;
  return [...cur, { filename: fn, label: guestLabel(fn) }];
}

export function removeGuestCheckpoint(guests, filename) {
  return (guests || []).filter((g) => g.filename !== filename);
}
