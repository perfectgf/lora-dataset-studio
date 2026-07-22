/** Several features are driven by a setting that is discoverable only by knowing it
 * exists — the manual Upscale & improve being the reported case: its strength and
 * instruction are editable, and nothing on the button said so. These links live
 * where the user is ACTING, not on the Settings page they are not on. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const read = (p) => fs.readFileSync(new URL(p, import.meta.url), 'utf8');
const link = read('./SettingsLink.jsx');
const registry = read('../settings/registry.js');

test('every section a link points at really exists in the settings registry', () => {
  const known = new Set([...registry.matchAll(/id: '([a-z0-9-]+)'/g)].map((m) => m[1]));
  assert.ok(known.size >= 8, 'settings registry did not parse');
  const files = [
    '../dataset/DatasetLightbox.jsx', '../dataset/CaptionToolsBar.jsx',
    '../dataset/TrainingPanel.jsx', '../dataset/ConceptSourcesPanel.jsx',
  ];
  let found = 0;
  for (const f of files) {
    for (const m of read(f).matchAll(/<SettingsLink section="([a-z0-9-]+)"/g)) {
      found += 1;
      assert.ok(known.has(m[1]), `${f} links to unknown settings section "${m[1]}"`);
    }
  }
  assert.ok(found >= 4, `expected links in every surveyed surface, found ${found}`);
});

test('a settings link never triggers the surface it sits on', () => {
  // These sit on top of tiles and lightboxes whose parents open viewers or start
  // jobs; without this a click would do both.
  assert.match(link, /onClick=\{\(e\) => e\.stopPropagation\(\)\}/);
});

test('the improve link is offered where the improve action is', () => {
  const lightbox = read('../dataset/DatasetLightbox.jsx');
  assert.match(lightbox, /Upscale & improve/);
  assert.match(lightbox, /<SettingsLink section="engines"/);
  // hidden while it runs — a settings trip mid-job is not the offer being made
  assert.match(lightbox, /\{onImprove && !improvementActive && \(/);
});

test('the cloud banner lands on the section holding the key, not the landing page', () => {
  const page = read('../../pages/CloudRunsPage.jsx');
  assert.match(page, /navigate\('\/settings\/training'\)/);
  assert.doesNotMatch(page, /navigate\('\/settings'\)/);
});
