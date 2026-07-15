import test from 'node:test';
import assert from 'node:assert/strict';
import { matchesQuery, SETTINGS_SECTIONS } from './registry.js';

test('small-image Klein rescue terms find Scraping & sources', () => {
  const scraping = SETTINGS_SECTIONS.find((section) => section.id === 'scraping');
  for (const query of ['klein', 'small image', 'rescue', 'upscale']) {
    assert.equal(matchesQuery(scraping, query), true, query);
  }
});
