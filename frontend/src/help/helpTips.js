/* One-time contextual tips. A tip is shown at most once EVER (per browser),
   tracked in localStorage['ldsHelpTipsSeen'] as a JSON map { topicId: true }.

   PURE JS: the persistence store is injectable so node --test can drive it with
   an in-memory fake. Tips are INDEPENDENT of Help mode — they appear whether or
   not the mode is on; that is their whole point. requestHelpTip() only fires an
   event; the TipHost (HelpMode.jsx) decides whether to actually show it. */

export const TIPS_SEEN_KEY = 'ldsHelpTipsSeen';
export const TIP_EVENT = 'lds:help-tip';

function defaultStore() {
  try {
    if (typeof localStorage !== 'undefined') return localStorage;
  } catch { /* access can throw in locked-down contexts */ }
  return null;
}

function readSeen(store) {
  const s = store || defaultStore();
  if (!s) return {};
  try {
    const parsed = JSON.parse(s.getItem(TIPS_SEEN_KEY) || '{}');
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch { return {}; }
}

function writeSeen(store, map) {
  const s = store || defaultStore();
  if (!s) return;
  try { s.setItem(TIPS_SEEN_KEY, JSON.stringify(map)); } catch { /* best-effort */ }
}

/** Has this tip never been shown yet? (Unknown topic → treat as showable.) */
export function shouldShowTip(topicId, store) {
  if (!topicId) return false;
  return !readSeen(store)[topicId];
}

/** Mark a tip as shown — forever. Idempotent. */
export function markTipSeen(topicId, store) {
  if (!topicId) return;
  const map = readSeen(store);
  if (map[topicId]) return;
  map[topicId] = true;
  writeSeen(store, map);
}

/** Best-effort: ask the TipHost to consider showing the tip for `trigger`.
    Never throws — instrumentation points call this without guarding. */
export function requestHelpTip(trigger) {
  if (!trigger) return;
  try {
    if (typeof window === 'undefined' || typeof window.dispatchEvent !== 'function') return;
    window.dispatchEvent(new CustomEvent(TIP_EVENT, { detail: { trigger } }));
  } catch { /* best-effort */ }
}
