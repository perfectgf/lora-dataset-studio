// Members previews: what an installation without an active, linked Patreon
// membership shows of the members features: a name, one sentence, a "Members"
// badge and a Join link to Patreon. A preview is text only. No code, stub,
// flag, route or path of a members feature ships with it, and a click opens
// nothing but the Patreon page.
//
// The server reads the signed list from the Patreon catalog (key
// `member_previews` of catalog.json) and answers GET /api/member-previews:
// nothing for an active, linked membership, nothing offline. This module
// checks every entry again before anything renders.
import { apiFetch } from '../../api/fetchClient.js';
import { MEMBERSHIP_CHANGED } from '../../plugins/membership.js';
import { safeJoinUrl } from './patreonModel.js';

export const PREVIEWS_FORMAT = 'lds-member-previews/1';
export const MAX_ITEMS = 40;
export const MAX_NAME = 40;
export const MAX_SUMMARY = 160;
const MAX_URL = 2048;
const ITEM_KEYS = new Set(['id', 'name', 'summary', 'placement', 'url']);
// A core members feature (krea_enhancer) or a plugin (local.creative_library).
const ID = /^[a-z][a-z0-9_]{1,31}(\.[a-z][a-z0-9_]{1,31})?$/;
// Controls, line separators and bidirectional overrides: a preview is plain text.
const UNSAFE_RANGES = [[0x00, 0x1f], [0x7f, 0x9f], [0x2028, 0x2029], [0x202a, 0x202e], [0x2066, 0x2069]];
const unsafeText = (value) => [...value].some((ch) => {
  const code = ch.codePointAt(0);
  return UNSAFE_RANGES.some(([low, high]) => code >= low && code <= high);
});
// Where this version shows previews. A newer catalog may name more places:
// this version leaves those entries out, one by one.
export const PLACEMENTS = Object.freeze([
  'studio.prompt', 'studio.options', 'training.options', 'dataset.crop', 'bank.crop', 'watermark.engines',
]);
// Test Studio options of one model family: studio.options.krea, studio.options.moss…
const FAMILY_PLACEMENT = /^studio\.options\.[a-z][a-z0-9]{1,23}$/;

function plainText(value, max) {
  if (typeof value !== 'string') return null;
  const trimmed = value.trim();
  return trimmed && trimmed.length <= max && !unsafeText(trimmed) ? trimmed : null;
}

function patreonUrl(value) {
  return typeof value === 'string' && value.length <= MAX_URL ? safeJoinUrl(value) : '';
}

export function knownPlacement(value) {
  return typeof value === 'string' && (PLACEMENTS.includes(value) || FAMILY_PLACEMENT.test(value));
}

/** `{items}` from a GET /api/member-previews answer. An entry with another
 *  key, a longer text, a link outside https://www.patreon.com or a place this
 *  version does not show is left out; another format gives no items. */
export function parseMemberPreviews(payload) {
  if (!payload || typeof payload !== 'object' || payload.format !== PREVIEWS_FORMAT
      || !Array.isArray(payload.items)) return { items: [] };
  const items = [];
  const seen = new Set();
  for (const item of payload.items.slice(0, MAX_ITEMS)) {
    if (!item || typeof item !== 'object' || Array.isArray(item)
        || Object.keys(item).some((key) => !ITEM_KEYS.has(key))) continue;
    const id = typeof item.id === 'string' && ID.test(item.id) ? item.id : null;
    const name = plainText(item.name, MAX_NAME);
    const summary = plainText(item.summary, MAX_SUMMARY);
    const url = patreonUrl(item.url);
    const key = `${item.placement}\u0000${id}`;
    if (!id || !name || !summary || !url || !knownPlacement(item.placement) || seen.has(key)) continue;
    seen.add(key);
    items.push({ id, name, summary, placement: item.placement, url });
  }
  return { items };
}

/** The previews shown at `placements`, in the order of the signed list. */
export function previewsAt(items, placements) {
  const wanted = new Set(placements);
  return items.filter((item) => wanted.has(item.placement));
}

let snapshot = { items: [] };
let version = 0;
let watching = false;
const listeners = new Set();

export function subscribeMemberPreviews(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export const memberPreviewsSnapshot = () => snapshot;

export async function refreshMemberPreviews(fetchPreviews = () => apiFetch('/api/member-previews', {
  cache: 'no-store', background: true,
})) {
  const current = ++version;
  let next = { items: [] };
  try {
    next = parseMemberPreviews(await fetchPreviews());
  } catch { /* offline or refused: nothing to show */ }
  if (current !== version) return;
  snapshot = next;
  for (const listener of listeners) listener();
}

/** Read the list once per page, and again whenever the membership changes
 *  (Connect, Disconnect, another tab). */
export function loadMemberPreviews(win = globalThis.window) {
  if (watching) return;
  watching = true;
  win?.addEventListener(MEMBERSHIP_CHANGED, () => { refreshMemberPreviews(); });
  win?.addEventListener('storage', (event) => { if (event.key === MEMBERSHIP_CHANGED) refreshMemberPreviews(); });
  refreshMemberPreviews();
}
