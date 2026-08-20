/**
 * One rule for every shot card: its tooltip says the PROMPT it will send.
 *
 * The panel renders three kinds of card from three separate blocks (built-in
 * catalog, 🔞 catalog, user-authored ✨/📥), and they had drifted into three
 * different behaviours: the user cards always showed the prompt, the 🔞 cards
 * showed it only until you generated something, and the built-in ones never
 * showed it at all. In both catalogs the prompt was being displaced by an
 * "N image(s) of this shot already in the dataset" message, which duplicated the
 * ✓×N badge sitting on the same card, so hovering the shots you had used most
 * told you the least. Found by a user, not by us.
 *
 * node --test cannot mount this JSX, so the rule is pinned by grep — the same
 * way grid-sort-contract pins the Sort wiring.
 */
import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const SRC = readFileSync(new URL('../src/components/dataset/VariationCatalog.jsx', import.meta.url), 'utf8')

test('every shot card tooltip carries the prompt, and nothing displaces it', () => {
  // The three cards, by the expression each one passes to `title`.
  assert.match(SRC, /title=\{e\.prompt\}/,
    'the catalog cards must show their prompt')
  assert.equal((SRC.match(/title=\{e\.prompt\}/g) || []).length, 2,
    'both catalogs (built-in and 🔞) must show it, not just one')
  assert.match(SRC, /title=\{blocked \? '🔞 shot[^']*' : c\.prompt\}/,
    'a user card shows its prompt, except when it is blocked and must say why')

  // The message that used to displace it must not come back as a title.
  assert.doesNotMatch(SRC, /title=\{done > 0 \?/,
    'the "already in the dataset" count must never be a tooltip again: it is on '
    + 'the card as ✓×N, and as a title it hides the prompt')
})

test('the ✓×N badge says what it means to a screen reader', () => {
  // Removing that sentence from the tooltips is only lossless if the badge
  // carries it. All three cards, not just the one that already did.
  assert.equal((SRC.match(/aria-label=\{`\$\{done\} already in the dataset`\}/g) || []).length, 3,
    'each of the three card kinds must label its count badge')
})
