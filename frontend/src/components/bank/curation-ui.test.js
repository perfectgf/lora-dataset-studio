// Reads the image Bank TREE, not one file: the Encre redesign split the
// workspace into a top bar, a filter rail, a passes panel and the grid, and a
// wiring assertion must survive a move (see bankTreeSource.js).
import { bankTreeSource } from './bankTreeSource.js';
import assert from 'node:assert/strict';
import test from 'node:test';

const ws = bankTreeSource();

// Regression guard for the reported symptom "🎨 Pick diverse and 🎯 Similar to
// selected show EXACTLY the same thing, whatever the reference". The backend
// selectors were always distinct and reference-sensitive; the bug was that the
// grid kept showing its default facet page and only scattered invisible
// checkmarks across a 24k-image bank, so BOTH buttons looked identical. The fix
// is that each selector feeds ITS OWN returned ids into a "show selected"
// (?ids=…) grid view. These assertions lock that wiring so a refactor can't
// collapse the two buttons back into the same, ref-insensitive view.

test('each curation button feeds its OWN returned ids into the selection view', () => {
  // Diverse posts to its endpoint and shows exactly what it got back …
  assert.match(ws, /\/api\/bank\/\$\{bankId\}\/select-diverse/);
  assert.match(ws, /showCuratedSelection\(d\.image_ids\)/);
  // … and Similar posts to a DIFFERENT endpoint, keyed on the selected image as
  // the reference, and shows ITS ranked ids — not the diverse ones.
  assert.match(ws, /\/api\/bank\/\$\{bankId\}\/select-similar/);
  assert.match(ws, /const ref = \[\.\.\.selected\]\[0\]/);
  assert.match(ws, /ref_id: ref/);
  // … and 🔤 Find by text is the third of the same family: its own endpoint, its
  // own ranked ids, the same view. (Its wording/limits contract lives in
  // bankTextSearch.test.js; here we only pin that it joins the family.)
  assert.match(ws, /\/api\/bank\/\$\{bankId\}\/search-text/);
  // … and ⚖️ Balanced pick is the fourth: its own endpoint, its own spread ids,
  // the same view. (Its readout contract lives in bankBalance.test.js.)
  assert.match(ws, /\/api\/bank\/\$\{bankId\}\/select-balanced/);
  // Every handler must route through the same "show what you selected" helper —
  // one call per selector. Bump this WITH the count when a fifth one lands; the
  // guard is that no two of them share a code path, not that there are exactly N.
  const feeds = ws.match(/showCuratedSelection\(d\.image_ids\)/g) || [];
  // Five: diverse, similar, ADD-MORE-similar (the same ranking, extended —
  // it re-shows the whole lot on purpose), text, balanced.
  assert.equal(feeds.length, 5, 'each selector feeds its own result into the view');
});

// "Most diverse" was computed as "most isolated" — the criterion that structurally
// prefers the meme, the stray photo of someone else and the botched frame. The
// backend now discounts isolation, and the UI must (a) send that setting, (b) let
// the user turn it back OFF (the historical behaviour is still one drag away), and
// (c) SAY what it does — a selector whose meaning changed silently would be worse
// than the bias it fixes.
test('🎨 Pick diverse exposes the typicality guard and sends it', () => {
  assert.match(ws, /const \[diverseTypicality, setDiverseTypicality\] = useState\(0\.5\)/);
  assert.match(ws, /typicality: diverseTypicality/);
  // a real, bounded control (0 → 1) the user can drag back to the old behaviour
  const slider = ws.match(/<input type="range"[^>]*setDiverseTypicality[^>]*\/>/s)
    || ws.match(/<input type="range" min=\{0\} max=\{1\}[\s\S]{0,240}?\/>/);
  assert.ok(slider, 'the guard is a slider, not a hidden constant');
  assert.match(slider[0], /min=\{0\} max=\{1\}/);
  // and it explains itself, including what OFF means
  assert.match(ws, /Skip the odd ones out/);
  assert.match(ws, /pure coverage, exactly like before/);
});

test('the curated selection actually switches the grid to a ?ids= view (not scattered checkmarks)', () => {
  // showCuratedSelection flips the grid into the selection view, seeded with the
  // ids in the order the backend ranked them, and drives the fetch immediately.
  assert.match(ws, /const showCuratedSelection = \(order\) => \{/);
  assert.match(ws, /setShowSelected\(true\)/);
  assert.match(ws, /refreshImages\(filter, 0, \{ on: true, order \}\)/);
  // refreshImages builds an id-scoped request from that order when the view is
  // on — this is what makes the selection VISIBLE instead of invisible ticks.
  assert.match(ws, /ids: \(order \|\| \[\]\)\.join\(','\)/);
});
