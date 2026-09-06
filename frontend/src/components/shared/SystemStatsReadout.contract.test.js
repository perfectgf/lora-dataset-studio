import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';

// A refusal is not a wall (2026-09-06): the arming itself lives in
// `maintenanceArming` and is exercised there (only a refusal with the matching
// flag arms; a double-click is not the second press; the offer expires). This
// contract pins that the readout goes THROUGH that module and nowhere else,
// sends the forced form for the job the offer named, and shows the armed state
// on the button for as long as the offer lives.
const src = readFileSync(new URL('./SystemStatsReadout.jsx', import.meta.url), 'utf8');

test('the readout arms only through the module, from the refusal body, and spends the arming on the next press', () => {
  assert.match(src, /import \{ OFFER_TOAST_MS, armFrom, armedUntil, consume \} from '\.\/maintenanceArming'/);
  assert.match(src, /const refused = \(kind, body\) => \{ if \(armFrom\(kind, body\)\) setArmedKind\(kind\); \}/);
  assert.match(src, /refused\('free', err\?\.body\)/);
  assert.match(src, /const forced = consume\('free'\)/);
  assert.doesNotMatch(src, /armed\.current/);
  assert.equal((src.match(/setArmedKind\(kind\)/g) || []).length, 1, 'the only place the armed state is set');
  assert.equal((src.match(/armFrom\(/g) || []).length, 1, 'armFrom is called from `refused` only');
});

test('the forced form names the job the offer named, on the same route as the first press', () => {
  assert.match(src, /const forcedBody = \(forced, flag\) => \(forced \? \{ \[flag\]: true, \.\.\.\(forced\.jobId \? \{ job_id: forced\.jobId \} : \{\}\) \} : \{\}\)/);
  assert.match(src, /postJson\('\/api\/system\/free-memory', forcedBody\(forced, 'interrupt'\)\)/);
});

test('the armed state is visible on the button for the life of the offer, and the toast says what the server did', () => {
  assert.match(src, /setTimeout\(\(\) => setArmedKind\(null\), Math\.max\(0, armedUntil\(armedKind\) - Date\.now\(\)\)\)/);
  assert.match(src, /data-armed=\{armedKind === 'free' \|\| undefined\}/);
  assert.match(src, /press again<\/span>/);
  assert.match(src, /aria-label=\{armedKind === 'free' \? "Free memory — press again to interrupt LDS's render" : 'Free memory'\}/);
  assert.match(src, /\(d\?\.interrupted \? 'The render was interrupted as asked\. ' : ''\) \+ freeMemorySummary\(d\)/);
  assert.match(src, /err\?\.body\?\.can_interrupt \? OFFER_TOAST_MS : undefined/);
});
