/** APP_VERSION only moves when a release is cut, so on a git checkout it names the
 *  last RELEASE and not the code being run — "You're up to date — v2026.07.21.1"
 *  while sitting twenty commits past it reads as a contradiction. */
import assert from 'node:assert/strict';
import test from 'node:test';

import { versionLabel } from './versionLabel.js';

test('a git checkout says which commit it is actually running', () => {
  assert.equal(
    versionLabel({ is_git: true, current: '2026.07.22.1', branch: 'main', current_sha: '4f8024c' }),
    'v2026.07.22.1 · main 4f8024c',
  );
});

test('a packaged install is unchanged — there the release version IS the truth', () => {
  assert.equal(versionLabel({ is_git: false, current: '2026.07.22.1' }), 'v2026.07.22.1');
  // no is_git field at all (release-tag path) behaves the same
  assert.equal(versionLabel({ current: '2026.07.22.1' }), 'v2026.07.22.1');
});

test('missing pieces degrade instead of printing holes', () => {
  // git checkout whose sha could not be read: fall back to the version alone
  assert.equal(versionLabel({ is_git: true, current: '2026.07.22.1' }), 'v2026.07.22.1');
  // no branch (detached HEAD): the sha alone still answers "what am I running"
  assert.equal(versionLabel({ is_git: true, current: '2026.07.22.1', current_sha: 'abc1234' }),
    'v2026.07.22.1 · abc1234');
  // no version at all: never render a bare "v"
  assert.equal(versionLabel({ is_git: true, current_sha: 'abc1234', branch: 'main' }), 'main abc1234');
  assert.equal(versionLabel({}), '');
  assert.equal(versionLabel(null), '');
});
