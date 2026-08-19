/* The announcement generator, and the credit detection it exposed.
 *
 * Every one of these cases is a real string from frontend/src/whatsNew.js, not
 * an invented one: the credit regex was verb-driven and MISSED ALL FIVE credits
 * of the 2026-07-28 wave, so several releases shipped with no Thanks line and
 * nobody noticed — a missing credit is silent by nature, which is exactly why
 * it needs a test rather than a reviewer.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { extractCredits } from './releaseNotes.mjs';
import { DISCORD_LIMIT, renderAnnouncement, renderLines, surfaceOf } from './discordAnnounce.mjs';

const entry = (id, title, blurb = 'x') => ({ id, title, blurb, date: '2026-07-28' });

test('every real credit form of the 28/07 wave is picked up', () => {
  const entries = [
    entry('a', 'A', 'The budget is in Settings. Thanks to j_o_e_l. (Discord) for the report.'),
    entry('b', 'B', 'Others were silent in the same way. Reported by 1Tomber (GitHub #23).'),
    entry('c', 'C', 'Works in WSL or Docker. Found and diagnosed by 1Tomber (GitHub #21).'),
    entry('d', 'D', 'Found, measured (~15 s on his install) and fixed by j_o_e_l. (Discord).'),
    entry('e', 'E', 'A clear win. Suggested by nofaceman (Reddit).'),
  ];
  assert.deepEqual(extractCredits(entries).sort(),
    ['1Tomber (GitHub)', 'j_o_e_l (Discord)', 'nofaceman (Reddit)']);
});

test('an issue number is a coordinate, not part of the name', () => {
  assert.deepEqual(extractCredits([entry('a', 'A', 'Reported by 1Tomber (GitHub #22).')]),
    ['1Tomber (GitHub)']);
});

test('prose that merely ends in a source name is not a credit', () => {
  /* The guard that replaced the verb list. A handle has no spaces; a sentence
     does. Without this, "…thanks to the work of everyone (Discord)" would
     credit a person who does not exist. */
  const entries = [
    entry('a', 'A', 'Announced to the whole community (Discord).'),
    entry('b', 'B', 'It now points to the release page (GitHub).'),
  ];
  assert.deepEqual(extractCredits(entries), []);
});

test('the same person credited twice is thanked once', () => {
  const entries = [
    entry('a', 'A', 'Reported by 1Tomber (GitHub #21).'),
    entry('b', 'B', 'Also found by 1Tomber (GitHub #22).'),
  ];
  assert.deepEqual(extractCredits(entries), ['1Tomber (GitHub)']);
});

test('announcing nothing throws instead of posting an empty message', () => {
  /* Silence is the defect: a wave that lists nothing is a wave that will be
     skipped, and a cheerful "0 changes" post is worse than no post. */
  assert.throws(() => renderAnnouncement({ tag: 'v1', entries: [], previousTag: 'v0' }),
    /nothing to announce/);
});

test('a single wave fits one message and keeps every entry', () => {
  const entries = Array.from({ length: 13 }, (_, i) => entry(`e${i}`, `Change number ${i}`));
  const [msg, ...rest] = renderAnnouncement({ tag: 'v1', entries, previousTag: 'v0' });
  assert.equal(rest.length, 0);
  assert.ok(msg.length <= DISCORD_LIMIT);
  for (const e of entries) assert.ok(msg.includes(e.title), `${e.title} was dropped`);
});

test('an oversized wave splits on entry boundaries, never mid-line', () => {
  const entries = Array.from({ length: 40 }, (_, i) =>
    entry(`e${i}`, `A fairly long headline about change number ${i} and what it gets you`));
  const parts = renderAnnouncement({ tag: 'v1', entries, previousTag: 'v0' });
  assert.ok(parts.length > 1, 'this wave should not have fitted one message');
  for (const p of parts) assert.ok(p.length <= DISCORD_LIMIT, `part is ${p.length} chars`);
  // No title may be cut in half, and none may go missing in the split.
  const joined = parts.join('\n');
  for (const e of entries) assert.ok(joined.includes(e.title), `${e.title} was dropped`);
  for (const line of renderLines(entries)) assert.ok(joined.includes(line));
});

test('a split announcement numbers its parts, thanks once, and opens once', () => {
  const entries = Array.from({ length: 40 }, (_, i) =>
    entry(`e${i}`, `A fairly long headline about change number ${i} and what it gets you`,
      i === 0 ? 'Reported by 1Tomber (GitHub #21).' : 'x'));
  const parts = renderAnnouncement({ tag: 'v1', entries, previousTag: 'v0' });
  assert.equal(parts.filter((p) => p.includes('is out —')).length, 1, 'one greeting');
  assert.equal(parts.filter((p) => p.includes('Thanks to')).length, 1, 'one thank-you');
  assert.ok(parts[0].includes('is out —'), 'the greeting opens part 1');
  assert.ok(parts.at(-1).includes('Thanks to'), 'the credits close the last part');
  parts.forEach((p, i) => assert.ok(p.includes(`part ${i + 1}/${parts.length}`)));
});

test('every line says WHERE the change lives, derived from the entry route', () => {
  // Titles say WHAT changed and never WHERE; a reader of #announcements does
  // not know which screen a line belongs to. The route the entry already
  // carries answers it, so nothing is re-typed and a moved route cannot leave
  // a stale label behind.
  const lines = renderLines([
    { title: 'Sweep for defects', to: '/video-bank' },
    { title: 'Crop in place', to: '/bank' },
    { title: 'Train from a preset', to: '/datasets?section=training' },
    { title: 'Pin a grid', to: '/canvas' },
  ]);
  assert.deepEqual(lines, [
    '• **Video bank** — Sweep for defects',
    '• **Bank** — Crop in place',
    '• **Datasets** — Train from a preset',
    '• **Canvas** — Pin a grid',
  ]);
});

test('/video-bank is never read as /bank, and an unknown route claims nothing', () => {
  // The longest-prefix rule is the whole reason '/video-bank' does not render
  // as 'Bank'; a route nobody mapped renders the bare title rather than
  // inventing a place for it.
  assert.equal(surfaceOf({ to: '/video-bank' }), 'Video bank');
  assert.equal(surfaceOf({ to: '/bank?filter=kept' }), 'Bank');
  assert.equal(surfaceOf({ to: '/somewhere-new' }), '');
  assert.equal(surfaceOf({}), '');
  assert.deepEqual(renderLines([{ title: 'Bare', to: '/nope' }]), ['• Bare']);
});
