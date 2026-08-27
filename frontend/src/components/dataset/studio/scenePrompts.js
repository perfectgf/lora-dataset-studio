/* 🎬 Scenes as a PROMPT BATCH — the two generation surfaces, the two sources.
 *
 * A bank of reference images and a dataset both carry one caption per image.
 * Read in row order those captions are a SEQUENCE (a storyboard, a chapter, a
 * shoot), and running them in that order with your own LoRA is a different
 * intent from the 🎲 random-caption shortcut, which stays what it is: ONE draw,
 * at random. A bank is the pile you triage; a dataset is what you KEPT — both
 * are legitimate sources of a sequence, so the panel offers either.
 *
 * Each ticked scene becomes one pass of the run's existing 📝 prompt axis — the
 * axis the prompt-history batch already rides (promptBatch.js), so the server
 * changes in nothing and the Test Studio and the board cannot drift.
 *
 * Plain .js (no JSX) so `node --test` executes all of it, worktree included.
 */

/** The two sources a scene can come from. The `kind` is what every rule below
 *  branches on, so a third source would land here and nowhere else. */
export const SCENE_SOURCES = [
  { kind: 'bank', label: '🗃 Bank', listUrl: '/api/banks', listKey: 'banks',
    pick: 'Choose a bank…', empty: 'No image bank yet' },
  { kind: 'dataset', label: '📁 Dataset', listUrl: '/api/dataset/list', listKey: 'datasets',
    pick: 'Choose a dataset…', empty: 'No dataset yet' },
];

/** The source descriptor a loaded payload becomes: ONE shape for both routes,
 *  so everything downstream (the summary line, the thumbnails) has one branch
 *  instead of two. `null` when the payload is not one this panel understands. */
export function sceneSource(kind, payload) {
  const d = payload || {};
  if (kind === 'dataset' && d.dataset_id != null) {
    return { kind: 'dataset', id: d.dataset_id, name: d.dataset_name || 'a dataset' };
  }
  if (kind === 'bank' && d.bank_id != null) {
    return { kind: 'bank', id: d.bank_id, name: d.bank_name || 'a bank' };
  }
  return null;
}

/** The URL of the picture a scene came from, or '' when there is none to show.
 *
 *  The two surfaces address a thumbnail differently — a bank by ROW ID, a
 *  dataset by FILE NAME — and a card that has neither (a bank from before
 *  thumbnails, a dataset image still rendering) must render WITHOUT an <img>
 *  rather than with one pointed at `/thumb/undefined`. */
export function sceneThumbUrl(source, scene) {
  const s = scene || {};
  if (!source || source.id == null) return '';
  if (source.kind === 'dataset') {
    return s.filename
      ? `/api/dataset/${source.id}/thumb/${encodeURIComponent(s.filename)}?s=256`
      : '';
  }
  if (source.kind === 'bank') {
    return s.image_id != null ? `/api/bank/${source.id}/thumb/${s.image_id}` : '';
  }
  return '';
}

/** One scene's FINAL prompt: the caption plus the ✏️ custom text typed on its
 *  card, or the caption alone when nothing was typed. The custom text is
 *  APPENDED — captions already open with the subject (and usually the trigger
 *  word), and what people add per scene is modifiers. The caption's trailing
 *  punctuation is dropped before the join so a captioner's closing period
 *  never yields "…on a bench., red dress". */
export function joinScenePrompt(prompt, extra) {
  const base = typeof prompt === 'string' ? prompt.trim() : '';
  const add = typeof extra === 'string' ? extra.trim() : '';
  if (!add) return base;
  if (!base) return add;
  return `${base.replace(/[\s,;.]+$/, '')}, ${add}`;
}

/** The prompts of the ticked scenes, in SCENE order — never in tick order.
 *  `picked` is a collection of indices into `scenes`; anything out of range or
 *  pointing at a promptless row is ignored rather than crashing the panel.
 *  `extras` maps a scene index to that card's ✏️ custom text; text typed on an
 *  UNTICKED scene changes nothing until the scene is ticked. */
export function scenePromptList(scenes, picked, extras) {
  const list = Array.isArray(scenes) ? scenes : [];
  const on = new Set(picked || []);
  const add = extras || {};
  return list
    .map((s, i) => {
      // The caption is the scene. Without one the row never runs — custom text
      // alone must not resurrect it, so the base is checked BEFORE the join.
      const base = s && typeof s.prompt === 'string' ? s.prompt.trim() : '';
      return on.has(i) && base ? joinScenePrompt(base, add[i]) : '';
    })
    .filter(Boolean);
}

/** Toggle one scene index in the picked collection (returns a NEW array). */
export function toggleSceneIndex(picked, index) {
  const cur = picked || [];
  return cur.includes(index) ? cur.filter((i) => i !== index) : [...cur, index];
}

/** The whole 📝 axis of a launch: history batch first (the older feature keeps
 *  its place), then the scenes in reading order. The server de-duplicates. */
export function combinedPromptBatch(historyPicked, scenes, picked, extras) {
  return [...(historyPicked || []), ...scenePromptList(scenes, picked, extras)];
}
