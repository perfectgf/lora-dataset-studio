/* The ⚖ compare dialog's pure half — everything `node --test` can execute.
 *
 * The JSX around it renders; THESE rules decide, so they live where a test can
 * call them (the same split as enhanceGate / classifyFramingGate). */

/** Which models start ticked: the current pick plus the next candidates, capped.
 *
 *  Capped at three because every candidate is a full 9B UNET swap through
 *  ComfyUI — tens of seconds each. Three is a judgement a person actually sits
 *  through; ticking all seven is a coffee break that ends in a grid nobody
 *  compares. Anything can still be ticked by hand. */
export const DEFAULT_COMPARE_CAP = 3;

export function defaultTicked(choices = [], stored = null) {
  const list = [];
  if (stored && choices.includes(stored)) list.push(stored);
  for (const m of choices) {
    if (list.length >= DEFAULT_COMPARE_CAP) break;
    if (!list.includes(m)) list.push(m);
  }
  return list;
}

export function toggleTicked(ticked, model) {
  return ticked.includes(model)
    ? ticked.filter((m) => m !== model)
    : [...ticked, model];
}

/** One seed per dialog OPEN, drawn once and reused for every candidate and
 *  every re-run — "same seed" is what makes the grid a comparison rather than
 *  a lottery. 32-bit on purpose: it travels through JSON untouched. */
export function compareSeed() {
  return Math.floor(Math.random() * 0xffffffff);
}
