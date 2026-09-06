/** The motion typed in the video studio, kept in this browser across a
 * reload (2026-09-06: refreshing the page emptied the one thing typed by
 * hand, while the Studio's own form has been refresh-safe for months). Read
 * once at mount, written on every change, best effort — a browser without
 * storage still renders the take. The text is kept as typed. */
export const PROMPT_STORAGE = 'lds.videoPromptDraft.v1';

export function readPromptDraft(storage) {
  try {
    storage ??= globalThis.localStorage;
    const saved = JSON.parse(storage?.getItem(PROMPT_STORAGE) || 'null');
    return typeof saved?.prompt === 'string' ? saved.prompt : '';
  } catch { return ''; }
}

export function writePromptDraft(prompt, storage) {
  try {
    storage ??= globalThis.localStorage;
    if (!storage) return;
    const text = typeof prompt === 'string' ? prompt : '';
    if (text) storage.setItem(PROMPT_STORAGE, JSON.stringify({ prompt: text }));
    else storage.removeItem(PROMPT_STORAGE);
  } catch { /* Storage disabled or full: the current take still works. */ }
}
