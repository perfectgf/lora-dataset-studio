import test from 'node:test';
import assert from 'node:assert/strict';
import { matchesQuery, SETTINGS_SECTIONS } from './registry.js';

test('small-image Klein rescue terms find Scraping & sources', () => {
  const scraping = SETTINGS_SECTIONS.find((section) => section.id === 'scraping');
  for (const query of ['klein', 'small image', 'rescue', 'upscale']) {
    assert.equal(matchesQuery(scraping, query), true, query);
  }
});

test('Vast offer filter terms find Training', () => {
  const training = SETTINGS_SECTIONS.find((section) => section.id === 'training');
  for (const query of ['verified', 'secure cloud', 'community cloud', 'offer filter']) {
    assert.equal(matchesQuery(training, query), true, query);
  }
});
