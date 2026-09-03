// ⚡ The Render panel's acceleration choice — the contract, read as text.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const read = (p) => readFileSync(join(here, p), 'utf8');
const panel = read('./VideoOptionsPanel.jsx');
const studio = read('./VideoTestStudio.jsx');
const setup = read('../../../../hooks/useSetupSteps.js');

test('the Turbo checkbox became a select over the arena podium, resolved by the server', () => {
  assert.doesNotMatch(panel, /set\(\{ turbo: v \}\)/, 'no checkbox posts a bare turbo flag any more');
  assert.match(panel, /data-testid="video-accel"/);
  assert.match(panel, /options\.accelerations : ACCELERATIONS/, 'the server list first, the static shape before it arrives');
  assert.match(panel, /disabled=\{a\.available === false\}/, 'a choice this machine cannot run is greyed');
  assert.match(panel, /Setup downloads it/, 'and says how to get it');
  assert.match(panel, /min-h-10 lg:min-h-0/, 'finger-sized on a phone');
});

test('the studio defaults to larryvrh, follows availability, reuses and reads back the name', () => {
  assert.match(studio, /accel: 'turbo', eros: false/);
  assert.match(studio, /pickAvailableAccel\(o\.accel, d\.accelerations\)/);
  assert.match(studio, /accel: clipAccel\(clip\)/, 'reuse restores the acceleration that made the clip');
  assert.doesNotMatch(studio, /turbo: !!clip\.turbo/);
});

test('Setup offers the two new weights beside larryvrh’s, in the Video Test Studio plan', () => {
  for (const id of ['h3_turbo_lora', 'h3_parasyte_lora', 'h3_dareties_lora']) {
    assert.match(setup, new RegExp(`${id}: 'Video acceleration:`), `${id} has a label`);
    assert.match(setup, new RegExp(`'${id}'`), `${id} is in the install plan`);
  }
});
