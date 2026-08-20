/**
 * ✨ Custom shots — the cards you write yourself in the generation panel.
 *
 * They live in localStorage (`datasetCustomShots`) as
 * {id, label, prompt, framing, nsfw}, and the label is DERIVED from the prompt,
 * never typed: a card shows the first characters of what you asked for, behind
 * the emoji of the register it rides. That derivation is the whole reason
 * editing needs rules of its own — change the prompt and the label has to
 * follow, or the card keeps advertising a shot it will no longer generate.
 *
 * Asked for by .samexit (Discord): a card could be added and deleted, never
 * corrected, so fixing one word of a 40-word prompt meant retyping all of it —
 * and the retyped card came back at the END of the row, unselected, which is not
 * what "edit" means to anyone.
 *
 * The SAME rules serve the 📥 Imported cards, because the line that matters is
 * not which group a card sits in: it is whether its name was DERIVED from its
 * prompt or CHOSEN by a human. A card promoted with ⇪ Keep carries a derived
 * name and lives on the server; a card from a JSON catalog carries a name its
 * author wrote. Re-deriving the second would delete the name. See
 * `hasDerivedLabel`.
 *
 * What an edit deliberately KEEPS:
 *  • the id — so the card stays where it is in the row, stays selected, and any
 *    preset that names it keeps naming it;
 *  • the register (`nsfw`) — an edit re-words a card, it does not re-file it.
 *    Deriving it from the CURRENT mode, like an add does, would silently demote
 *    a 🔞 card to ✨ because the user happened to fix a typo with the mode off;
 *  • a name a human chose. Only a name the app itself derived follows the prompt.
 *
 * What it deliberately does NOT keep: the ✓×N tally on the card. That badge
 * counts images stamped with the OLD label, and they were generated from the old
 * words — carrying the number over would credit this prompt with pictures it
 * never made.
 *
 * Pure functions over plain arrays: the component owns the state and the
 * storage, this file owns the rules.
 */

/** How much of the prompt the card shows. Stored inside every label ever
 *  written, so it is a value with history: raising it does not retroactively
 *  lengthen the labels already in localStorage, and lowering it does not
 *  shorten them — only an edit re-derives one. */
export const CUSTOM_SHOT_PREVIEW_CHARS = 40;

const FRAMINGS = ['face', 'bust', 'body', 'back'];

/** The visible name of a custom card: register emoji + the head of the prompt.
 *  The single place that derivation exists, so an add and an edit can never
 *  disagree about what a card is called. */
export function customShotLabel(prompt, nsfw = false) {
  const head = String(prompt || '').trim().slice(0, CUSTOM_SHOT_PREVIEW_CHARS);
  return `${nsfw ? '🔞' : '✨'} ${head}`;
}

/** A brand new card. `id` is injectable because tests (and any future undo) need
 *  a stable one — the default keeps the historical `custom_<epoch>` shape that
 *  stored presets and stored selections already contain. */
export function createCustomShot({ prompt, framing, nsfw = false, id = null }) {
  const text = String(prompt || '').trim();
  if (!text) return null;
  const shot = {
    id: id || `custom_${Date.now()}`,
    label: customShotLabel(text, nsfw),
    prompt: text,
    framing: FRAMINGS.includes(framing) ? framing : 'body',
  };
  return nsfw ? { ...shot, nsfw: true } : shot;
}

/** Is this card's name the app's own derivation of its prompt, or a name a
 *  person wrote? The ✨ cards (and anything ⇪ Keep promoted from one) carry a
 *  derived name and must follow their prompt. An imported catalog entry carries
 *  its author's name, and an edit that silently rewrote it would be a small act
 *  of vandalism on the one part of the card the user actually typed. */
export function hasDerivedLabel(shot) {
  return !!shot && shot.label === customShotLabel(shot.prompt, !!shot.nsfw);
}

/**
 * Re-word an existing card IN PLACE.
 *
 * Returns the SAME array reference when nothing can change (unknown id, empty
 * prompt, or a draft identical to what is already stored) — the caller stores
 * this in React state, and handing back a new array for a no-op would rerender
 * the whole panel on every keystroke of a cancelled edit.
 */
export function editCustomShot(shots, id, { prompt, framing }) {
  const list = shots || [];
  const index = list.findIndex((s) => s && s.id === id);
  if (index < 0) return list;
  const current = list[index];
  const text = String(prompt || '').trim();
  if (!text) return list;
  const nextFraming = FRAMINGS.includes(framing) ? framing : current.framing;
  if (text === current.prompt && nextFraming === current.framing) return list;
  const nsfw = !!current.nsfw;
  const next = [...list];
  next[index] = {
    ...current,
    label: hasDerivedLabel(current) ? customShotLabel(text, nsfw) : current.label,
    prompt: text,
    framing: nextFraming,
  };
  return next;
}

/** The draft the editor opens with, or null when the card is gone. Keeps the
 *  component from reaching into the shot shape in three places. */
export function customShotDraft(shots, id) {
  const shot = (shots || []).find((s) => s && s.id === id);
  return shot ? { prompt: shot.prompt, framing: shot.framing } : null;
}
