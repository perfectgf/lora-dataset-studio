/* 🎬 Scenes from a bank as a PROMPT BATCH — the two generation surfaces.
 *
 * A bank of reference images already carries one caption per image. Read in
 * bank order those captions are a SEQUENCE (a storyboard, a chapter, a shoot),
 * and running them in that order with your own LoRA is a different intent from
 * the 🎲 random-caption shortcut, which stays what it is: ONE draw, at random.
 *
 * Each ticked scene becomes one pass of the run's existing 📝 prompt axis — the
 * axis the prompt-history batch already rides (promptBatch.js), so the server
 * changes in nothing and the Test Studio and the board cannot drift.
 *
 * Plain .js (no JSX) so `node --test` executes all of it, worktree included.
 */

/** The prompts of the ticked scenes, in SCENE order — never in tick order.
 *  `picked` is a collection of indices into `scenes`; anything out of range or
 *  pointing at a promptless row is ignored rather than crashing the panel. */
export function scenePromptList(scenes, picked) {
  const list = Array.isArray(scenes) ? scenes : [];
  const on = new Set(picked || []);
  return list
    .map((s, i) => (on.has(i) ? (s && typeof s.prompt === 'string' ? s.prompt.trim() : '') : ''))
    .filter(Boolean);
}

/** Toggle one scene index in the picked collection (returns a NEW array). */
export function toggleSceneIndex(picked, index) {
  const cur = picked || [];
  return cur.includes(index) ? cur.filter((i) => i !== index) : [...cur, index];
}

/** The whole 📝 axis of a launch: history batch first (the older feature keeps
 *  its place), then the scenes in reading order. The server de-duplicates. */
export function combinedPromptBatch(historyPicked, scenes, picked) {
  return [...(historyPicked || []), ...scenePromptList(scenes, picked)];
}
