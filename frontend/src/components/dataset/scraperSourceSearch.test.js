import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildPexelsSearchUrl,
  isPexelsUrl,
  loadPexelsAuthorization,
  normalizePexelsKeyword,
  resolveScanTarget,
  savePexelsAuthorization,
} from './scraperSourceSearch.js';

function memoryStorage() {
  const values = new Map();
  return { getItem: (key) => values.has(key) ? values.get(key) : null,
    setItem: (key, value) => values.set(key, String(value)), removeItem: (key) => values.delete(key) };
}

test('Pexels keywords normalize slash-like separators and whitespace', () => {
  assert.equal(normalizePexelsKeyword('  portrait\\cinematic / nuit\t bleue '),
    'portrait cinematic nuit bleue');
});

test('Pexels search URLs encode the keyword and use the selected locale', () => {
  assert.equal(buildPexelsSearchUrl('portrait été', 'fr-FR'),
    'https://www.pexels.com/fr-fr/chercher/portrait%20%C3%A9t%C3%A9/');
  assert.equal(buildPexelsSearchUrl('film portrait', 'en-US', 'landscape'),
    'https://www.pexels.com/en-us/search/film%20portrait/?orientation=landscape');
  assert.equal(buildPexelsSearchUrl('été & nuit ? #50% / studio', 'fr-FR'),
    'https://www.pexels.com/fr-fr/chercher/%C3%A9t%C3%A9%20%26%20nuit%20%3F%20%2350%25%20studio/');
});

test('unsupported Pexels orientation is never added to the generated URL', () => {
  assert.equal(buildPexelsSearchUrl('portrait', 'fr-FR', 'wide'),
    'https://www.pexels.com/fr-fr/chercher/portrait/');
  assert.equal(buildPexelsSearchUrl(' / \\ ', 'fr-FR'), '');
});

test('pagination stays pinned to the active scan instead of an edited draft', () => {
  const values = {
    explicitUrl: undefined,
    draftUrl: 'https://example.test/edited',
    activeScanUrl: 'https://example.test/scanned',
  };
  assert.equal(resolveScanTarget({ ...values, nextPage: 0 }), values.draftUrl);
  assert.equal(resolveScanTarget({ ...values, nextPage: 3 }), values.activeScanUrl);
  assert.equal(resolveScanTarget({ ...values, nextPage: 0,
    explicitUrl: ' https://www.reddit.com/search/?q=portrait ' }),
  'https://www.reddit.com/search/?q=portrait');
});

test('Pexels dataset authorization is global and removable', () => {
  const storage = memoryStorage();
  assert.equal(loadPexelsAuthorization(storage), false);
  savePexelsAuthorization(true, storage);
  assert.equal(loadPexelsAuthorization(storage), true);
  savePexelsAuthorization(false, storage);
  assert.equal(loadPexelsAuthorization(storage), false);
});

test('Pexels URL detection accepts only official web hosts over HTTP or HTTPS', () => {
  assert.equal(isPexelsUrl('https://www.pexels.com/photo/portrait-12/'), true);
  assert.equal(isPexelsUrl('https://pexels.com/collections/editorial-abc/'), true);
  assert.equal(isPexelsUrl('http://www.pexels.com/search/x/'), true);
  assert.equal(isPexelsUrl('ftp://www.pexels.com/search/x/'), false);
  assert.equal(isPexelsUrl('https://www.pexels.com.evil.test/search/x/'), false);
  assert.equal(isPexelsUrl('https://pexels.com@evil.test/search/x/'), false);
});
