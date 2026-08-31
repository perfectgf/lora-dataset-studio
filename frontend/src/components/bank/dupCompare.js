/** ⤢ Compare — picking the keeper of a duplicate / "same shot" group at a size
 * where the choice can actually be made.
 *
 * WHY THIS EXISTS. The resolution panel offered three ways to settle a group:
 * *Keep best*, *Keep first*, or clicking one of its 96-pixel thumbnails. The
 * first two are bulk verdicts you take on trust; the third asks you to tell two
 * near-identical shots apart in a stamp. So the honest answer to "is that the
 * right copy?" was: you cannot see it from here. That is the gap this lane
 * closes — the same images, full screen, side by side, with the numbers that
 * separate them and one keystroke per verdict.
 *
 * PURE, like `bankReview.js` next door and for the same reason: `node --test`
 * parses no JSX, so what is worth being sure about — where the cursor lands,
 * what a decision does to the queue, which keystroke means what — lives here and
 * the component stays a renderer.
 *
 * ── The four decisions this file makes ───────────────────────────────────────
 *
 * 1. THE QUEUE UNIT IS THE GROUP, THE CURSOR UNIT IS THE COPY. ▶ Review walks
 *    images; this walks GROUPS, and inside each one a cursor sits on one copy.
 *    That is the whole difference between the two surfaces and it is what makes
 *    the arrows mean something else here (see `dupCompareKeyAction`).
 *
 * 2. THE CURSOR OPENS ON THE APP'S OWN PICK. `best_id` is where the run starts,
 *    so K is "yes, that one" and moving off it is a deliberate disagreement.
 *    Opening on the first member would make the recommendation a thing you have
 *    to go and find, which is how it ends up ignored.
 *
 * 3. A SKIPPED GROUP NEVER COMES BACK IN THIS SESSION. Skip writes nothing to
 *    the database, so the server still calls the group unresolved and hands it
 *    back on the next page fetch. Same doctrine as `bankReview.skip`: "not now"
 *    is honoured for the length of the run, or the refill turns into a loop.
 *
 * 4. THE LIST IS REFILLED, NOT PAGED. Resolving a group drops it out of the
 *    server's unresolved set, so offsets shift under the walk. Refilling from
 *    offset 0 and dropping the gids already seen is the only paging that stays
 *    true while the thing being paged is shrinking.
 */
import { ownsTypedKeys, reviewKeyAction } from '../shared/reviewShortcuts.js';
import { railIsColumn } from './bankLayout.js';

/** Which layout a run opens in. Side by side needs room for two pictures at a
 * useful size; below the width where the filter rail itself stops fitting
 * beside the grid, three copies in a row are three stamps again — which is the
 * exact problem this screen exists to fix. So a narrow screen opens on the full
 * frame and flips between copies with ← →. Reuses the rail's own breakpoint
 * rather than inventing a second one for the same question. */
export function startingLayout(width) {
  return railIsColumn(width) ? 'side' : 'single';
}

/* ── Keys ─────────────────────────────────────────────────────────────────── */

/** What the footer prints, and the only list of this lane's shortcuts. The
 * one-line hint is derived from it, so what the UI PROMISES cannot drift from
 * what the handler does. */
export const COMPARE_SHORTCUTS = [
  { keys: 'K', what: 'Keep the copy under the cursor — the rest of the group is rejected' },
  { keys: 'R', what: 'Reject this copy only, and move to the next one' },
  { keys: 'N', what: 'Not duplicates — keep every copy and never propose this group again' },
  { keys: 'B', what: 'Put the cursor back on the app’s pick (BEST)' },
  { keys: '←  →', what: 'Move between the copies of this group' },
  { keys: '1 … 9', what: 'Jump straight to a copy' },
  { keys: 'S', what: 'Skip this group — decides nothing, not shown again in this run' },
  { keys: '⇧←  ⇧→', what: 'Previous / next group' },
  { keys: 'F', what: 'Switch between side by side and full screen' },
  { keys: 'Esc', what: 'Leave' },
];

export const COMPARE_HINT = 'K keep this copy · R reject it · N not duplicates'
  + ' · B back to the app’s pick · ← → between copies · 1-9 jump straight to one'
  + ' · S skip the group · ⇧← ⇧→ move between groups · F full screen · Esc leave';

/**
 * What this keystroke means: 'keep' | 'reject' | 'best' | 'skip' | 'layout'
 * | 'prev-member' | 'next-member' | 'prev-group' | 'next-group' | 'close' | null.
 *
 * K, R, S and Esc are read off the SHARED grammar (`shared/reviewShortcuts.js`)
 * — the same four letters the Bank's ▶ Review and the dataset lightbox obey, so
 * a reflex learnt on one screen is right on this one.
 *
 * THE ARROWS ARE DELIBERATELY RE-POINTED, and this is the one place in the app
 * where they are. In a queue lightbox → is "move on without judging"; here
 * moving on means leaving the GROUP, and the thing your eye actually wants to do
 * is flip between the copies of the one on screen. So the bare arrows walk the
 * copies, ⇧ walks the groups, and S keeps the shared "move on without judging"
 * meaning at the level this surface decides at. Written down rather than
 * discovered: an arrow that quietly means something else on one screen is worse
 * than an arrow that means something else on purpose.
 */
export function dupCompareKeyAction(event) {
  if (!event) return null;
  if (event.metaKey || event.ctrlKey || event.altKey) return null;
  // Answered before the typing guard, like the shared grammar does: a field
  // focused inside an overlay must never trap the user in it.
  if (event.key === 'Escape') return 'close';
  if (ownsTypedKeys(event.target)) return null;
  const key = typeof event.key === 'string' ? event.key : '';
  if (key === 'ArrowLeft') return event.shiftKey ? 'prev-group' : 'prev-member';
  if (key === 'ArrowRight') return event.shiftKey ? 'next-group' : 'next-member';
  if (event.shiftKey) return null;
  const shared = reviewKeyAction(event);
  if (shared === 'keep' || shared === 'reject' || shared === 'skip') return shared;
  const letter = key.toLowerCase();
  if (letter === 'n') return 'distinct';
  if (letter === 'b') return 'best';
  if (letter === 'f') return 'layout';
  return null;
}

/** 1-9 → a 0-based copy index, or null. Its own function rather than a value
 * smuggled through `dupCompareKeyAction`'s string return: two questions, two
 * answers, both testable. */
export function memberKeyIndex(event) {
  if (!event || event.metaKey || event.ctrlKey || event.altKey || event.shiftKey) return null;
  if (ownsTypedKeys(event.target)) return null;
  const key = typeof event.key === 'string' ? event.key : '';
  if (!/^[1-9]$/.test(key)) return null;
  return Number(key) - 1;
}

/* ── The session ──────────────────────────────────────────────────────────── */

const groupId = (g) => (g && g.group != null ? g.group : null);

/** Where the cursor goes when a group opens: the app's pick, or the first copy
 * still standing. Never a rejected one — the run would open on a decision
 * already taken. */
function cursorFor(group, rejected) {
  const images = (group && group.images) || [];
  const live = images.filter((im) => !isRejected(im, rejected));
  const pick = live.find((im) => im.id === group?.best_id) || live[0] || images[0];
  const i = images.indexOf(pick);
  return i < 0 ? 0 : i;
}

/** Rejected HERE (this session) or already rejected in the bank. Both count:
 * a group re-opened after a partial resolve must not offer a dead copy again. */
export function isRejected(img, rejected = []) {
  if (!img) return true;
  return img.status === 'reject' || rejected.includes(img.id);
}

/** Open a compare run over the groups the panel is showing.
 *
 * `startGroup` is the card the user clicked; it opens THERE and walks forward,
 * exactly like ▶ Review's `startId`. An unknown id simply starts at the top
 * rather than emptying the run. */
export function createCompare(groups, opts = {}) {
  const { startGroup = null } = opts;
  const list = (Array.isArray(groups) ? groups : []).filter((g) => groupId(g) != null);
  let gi = 0;
  if (startGroup != null) {
    const at = list.findIndex((g) => g.group === startGroup);
    if (at >= 0) gi = at;
  }
  return {
    groups: list,
    gi,
    mi: cursorFor(list[gi], []),
    seen: list.map(groupId),
    resolved: [],
    // Groups answered with ≠ ("not duplicates"). Kept apart from `resolved`
    // because they are the opposite outcome — nothing was rejected — and the
    // end-of-run screen has to be able to say which of the two it is doing.
    vetoed: [],
    skipped: [],
    rejected: [],
  };
}

export function currentGroup(s) {
  return s && s.gi >= 0 && s.gi < s.groups.length ? s.groups[s.gi] : null;
}

export function isExhausted(s) {
  return !s || s.gi >= s.groups.length;
}

/** The copies of the current group, rejected ones included — they stay ON
 * SCREEN, greyed, because "I already threw that one out" is half of what makes
 * the remaining choice readable. */
export function currentImages(s) {
  const g = currentGroup(s);
  return (g && g.images) || [];
}

export function currentMember(s) {
  const images = currentImages(s);
  return images[s?.mi] || null;
}

/** How many copies of this group are still standing. Two is a choice, one is a
 * group that has resolved itself and can be left behind. */
export function liveCount(s) {
  return currentImages(s).filter((im) => !isRejected(im, s.rejected)).length;
}

/** Move the cursor to another copy. Wraps inside the group and never lands on a
 * rejected copy — a wrap that stops on a dead one reads as a stuck key. */
export function moveMember(s, delta) {
  const images = currentImages(s);
  const n = images.length;
  if (n === 0) return s;
  for (let step = 1; step <= n; step += 1) {
    const at = (((s.mi + delta * step) % n) + n) % n;
    if (!isRejected(images[at], s.rejected)) return { ...s, mi: at };
  }
  return s;
}

/** 1-9 and a click both land here. Out of range or already rejected: ignored,
 * rather than moved somewhere the user did not ask for. */
export function pickMember(s, index) {
  const images = currentImages(s);
  if (!Number.isInteger(index) || index < 0 || index >= images.length) return s;
  if (isRejected(images[index], s.rejected)) return s;
  return { ...s, mi: index };
}

/** Put the cursor back on the app's pick (B). */
export function pickBest(s) {
  const g = currentGroup(s);
  if (!g) return s;
  return { ...s, mi: cursorFor(g, s.rejected) };
}

function seatAt(s, gi) {
  return { ...s, gi, mi: cursorFor(s.groups[gi], s.rejected) };
}

/** Forward one group. Stops past the end — the shell asks for a refill there
 * rather than wrapping, so nothing is ever re-proposed silently. */
export function nextGroup(s) {
  return isExhausted(s) ? s : seatAt(s, s.gi + 1);
}

export function prevGroup(s) {
  return s.gi <= 0 ? s : seatAt(s, s.gi - 1);
}

/** "Not now." Nothing is written; the group is remembered so a refill cannot
 * hand it back in this run. */
export function skipGroup(s) {
  const gid = groupId(currentGroup(s));
  if (gid == null) return nextGroup(s);
  const skipped = s.skipped.includes(gid) ? s.skipped : [...s.skipped, gid];
  return nextGroup({ ...s, skipped });
}

/** The server accepted a resolve on this group — record it and move on. Callers
 * must not call this on a failed POST: the group has to stay under the cursor
 * with the error visible, exactly as `bankReview.decide` demands.
 *
 * `keptId` is the copy the server just ELECTED, and passing it matters: the
 * server rejected every other member in the same call, and the session has to
 * know. ⇧← walks back into a settled group, and without this its tiles still
 * look alive — a second K there sends keep_ids for a copy whose rivals are
 * already rejected, `resolve_dups` rejects the lone survivor too, and the whole
 * group ends up empty. Recording the losers greys them out and disarms the
 * button instead. Omitted by `rejectMember`, which resolves a group by
 * elimination and must leave the survivor standing. */
export function resolveGroup(s, gid = null, keptId = null) {
  const id = gid == null ? groupId(currentGroup(s)) : gid;
  if (id == null) return s;
  const resolved = s.resolved.includes(id) ? s.resolved : [...s.resolved, id];
  let rejected = s.rejected;
  if (keptId != null) {
    const losers = currentImages(s)
      .filter((im) => im.id !== keptId && !isRejected(im, rejected))
      .map((im) => im.id);
    if (losers.length) rejected = [...rejected, ...losers];
  }
  return nextGroup({ ...s, resolved, rejected });
}

/** ≠ — the server accepted "these are not duplicates". The group is answered
 * WITHOUT anything being rejected, and it will not be proposed again (the
 * refill drops it, like a skip, and the server stops listing it at all).
 *
 * Same rule as `resolveGroup`: never call this on a failed POST. */
export function vetoGroup(s, gid = null) {
  const id = gid == null ? groupId(currentGroup(s)) : gid;
  if (id == null) return s;
  const vetoed = s.vetoed.includes(id) ? s.vetoed : [...s.vetoed, id];
  return nextGroup({ ...s, vetoed });
}

/** One copy rejected on its own (R, or the ✕ under a tile). The group stays open
 * while two copies are still standing; when only one is left there is no choice
 * to make any more, so the run treats it as resolved and moves on. */
export function rejectMember(s, imageId) {
  if (imageId == null) return s;
  const rejected = s.rejected.includes(imageId) ? s.rejected : [...s.rejected, imageId];
  const next = { ...s, rejected };
  const live = currentImages(next).filter((im) => !isRejected(im, rejected));
  if (live.length <= 1) return resolveGroup(next);
  // The cursor moves only when the copy it was SITTING ON is the one that just
  // went. Side by side, the ✕ under a tile rejects THAT tile — moving the cursor
  // off a picture the user did not touch is how the next keystroke lands on the
  // wrong one.
  const under = currentImages(next)[next.mi];
  if (!under || under.id !== imageId) return next;
  return moveMember(next, 1);
}

/** Append what a refill fetch returned, minus everything this run has already
 * walked, skipped or settled. Returns the SAME object when nothing is new, so
 * the shell can tell "there is more" from "that is all there is". */
export function appendGroups(s, groups) {
  const fresh = (Array.isArray(groups) ? groups : []).filter((g) => {
    const gid = groupId(g);
    return gid != null && !s.seen.includes(gid) && !s.skipped.includes(gid)
      && !s.resolved.includes(gid) && !s.vetoed.includes(gid);
  });
  if (!fresh.length) return s;
  const next = {
    ...s,
    groups: [...s.groups, ...fresh],
    seen: [...s.seen, ...fresh.map(groupId)],
  };
  // A refill that arrives while the cursor sits past the end seats it on the
  // first new group; one that arrives mid-run must not move the cursor at all.
  return isExhausted(s) ? seatAt(next, s.groups.length) : next;
}

/** The honest readout. `position` counts the groups this run has WALKED, not
 * the bank's unresolved total — that total shrinks with every resolve, and a
 * denominator that moves under a progress bar is worse than no bar. */
export function compareProgress(s) {
  return {
    position: Math.min(s.gi + 1, s.groups.length),
    loaded: s.groups.length,
    resolved: s.resolved.length,
    vetoed: s.vetoed.length,
    skipped: s.skipped.length,
    rejected: s.rejected.length,
  };
}

/* ── Reading the copies against each other ────────────────────────────────── */

const MB = 1024 * 1024;

/** The four numbers that separate two copies of one shot, in the order they
 * settle an argument. Resolution and sharpness are the pair that decides most
 * duplicate groups; the aesthetic score only exists once ✨ Score has run, and
 * file size is the tie-break that catches a re-compressed copy at identical
 * dimensions. */
const METRICS = [
  { key: 'aesthetic', label: 'aesthetic', of: (im) => im.aesthetic_score,
    text: (v) => v.toFixed(1) },
  { key: 'pixels', label: 'resolution', of: (im) => ((im.width || 0) * (im.height || 0)) || null,
    text: (_v, im) => `${im.width || '?'}×${im.height || '?'}` },
  { key: 'sharp', label: 'sharpness', of: (im) => im.blur_score,
    text: (v) => `sharpness ${Math.round(v)}` },
  { key: 'bytes', label: 'file size', of: (im) => im.file_size,
    text: (v) => (v >= MB ? `${(v / MB).toFixed(1)} MB` : `${Math.round(v / 1024)} kB`) },
];

/** Per-copy facts, each one lit when the copy holds the group's TOP value.
 *
 * The lit chip is what makes a group readable at a glance: three copies, and
 * the one with no chips lit is the one that loses on everything — which is the
 * sentence the user was trying to compose out of thumbnails. A metric no copy
 * carries is absent rather than printed as '?'.
 *
 * TIES LIGHT EVERY COPY THAT HOLDS THE TOP, not nobody. The strict rule was
 * tried first and it goes blank on the commonest group of all — one file and
 * its exact copy, plus a re-compressed third — because the two identical copies
 * tie on every metric. Blank chips there read as "nothing measured", when what
 * is true is "these two are top and the third is worse". Whether the PICK
 * deserves credit for a tie is a different question, and `bestWins` answers it
 * strictly. */
export function compareFacts(images, img) {
  const rows = Array.isArray(images) ? images : [];
  if (!img) return [];
  return METRICS.map((m) => {
    const v = m.of(img);
    if (v == null) return null;
    const values = rows.map((r) => m.of(r)).filter((x) => x != null);
    const top = values.length ? Math.max(...values) : null;
    return { key: m.key, label: m.label, text: m.text(v, img), win: top != null && v === top };
  }).filter(Boolean);
}

/** Which copies are byte-for-byte the same picture, as 1-based positions.
 *
 * In a duplicate group this is the most useful fact on the screen and the one
 * the resolution panel could never say: *these two are the same file, keep
 * either — the third is the odd one out*. Judged on the facts the row already
 * carries (dimensions and weight) over images the dHash pass has ALREADY proved
 * are the same shot, so equal dimensions plus equal bytes is identity, not a
 * coincidence. A row with no weight yet is not compared at all — "0 = 0" would
 * declare a whole unscanned group identical. */
export function twinPositions(images) {
  const rows = Array.isArray(images) ? images : [];
  const sig = (im) => `${im?.width || 0}x${im?.height || 0}:${im?.file_size || 0}`;
  const out = {};
  rows.forEach((im, i) => {
    if (!im || !im.file_size || !im.width || !im.height) return;
    const mine = sig(im);
    const others = rows.reduce((acc, r, j) => (
      j !== i && sig(r) === mine ? [...acc, j + 1] : acc), []);
    if (others.length) out[im.id] = others;
  });
  return out;
}

/** What the elected copy actually wins on, as plain words.
 *
 * DERIVED FROM THE NUMBERS ON SCREEN, never from a second copy of the server's
 * election rule (`image_bank_service._best_of`). A frontend that re-implemented
 * that ordering would drift the day the ordering changes and would then explain
 * a choice nobody made. Saying only "it is bigger and sharper" — which is
 * checkable against the chips right next to it — cannot go stale. */
export function bestWins(images, bestId) {
  const rows = Array.isArray(images) ? images : [];
  const best = rows.find((r) => r.id === bestId);
  if (!best) return [];
  return METRICS.filter((m) => {
    const v = m.of(best);
    if (v == null) return false;
    return rows.every((r) => r.id === bestId || m.of(r) == null || m.of(r) < v);
  }).map((m) => m.label);
}

/** The BEST badge's tooltip: what it wins on, or an honest admission that the
 * copies are indistinguishable by anything measured. */
export function bestReasonText(images, bestId) {
  const wins = bestWins(images, bestId);
  if (!wins.length) {
    return 'The app’s pick. Nothing measured separates these copies — the'
      + ' tie-break is import order, so this is a coin toss you may want to make'
      + ' yourself.';
  }
  return `The app’s pick — best of the group on ${wins.join(' and ')}.`;
}
