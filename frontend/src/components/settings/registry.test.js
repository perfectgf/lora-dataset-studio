import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { matchesQuery } from './registry.js';

import scrape from '../../../../bundled/scrape/frontend/index.js';
import cloud from '../../../../bundled/cloud_training/frontend/index.js';

test('small-image Klein rescue terms find the Web scraping plugin settings', () => {
  const scraping = scrape.slots['settings.group'].find(group => group.id === 'web-scraping');
  for (const query of ['klein', 'small image', 'rescue', 'upscale']) {
    assert.equal(matchesQuery(scraping, query), true, query);
  }
});

test('Pexels API credential terms find the Web scraping plugin settings', () => {
  const scraping = scrape.slots['settings.group'].find(group => group.id === 'web-scraping');
  for (const query of ['pexels', 'pexels api', 'api key', 'quota']) {
    assert.equal(matchesQuery(scraping, query), true, query);
  }
});

test('Pexels key and attribution markup stay wired without nested controls', () => {
  const settingsSource = readFileSync(new URL('../../../../bundled/scrape/frontend/panels/ScrapeSettingsGroup.jsx', import.meta.url), 'utf8').replace(/\r\n/g, '\n');
  const panelSource = readFileSync(
    new URL("../../../../bundled/scrape/frontend/panels/ConceptSourcesPanel.jsx", import.meta.url), 'utf8').replace(/\r\n/g, '\n');
  const scraperSourceSearchSource = readFileSync(
    new URL("../../../../bundled/scrape/frontend/lib/scraperSourceSearch.js", import.meta.url), 'utf8').replace(/\r\n/g, '\n');
  const attributionSource = readFileSync(
    new URL('../dataset/PexelsAttribution.jsx', import.meta.url), 'utf8').replace(/\r\n/g, '\n');
  const readmeSource = readFileSync(new URL('../../../../README.md', import.meta.url), 'utf8').replace(/\r\n/g, '\n');
  const installationSource = readFileSync(new URL('../../../../docs/guide/installation.md', import.meta.url), 'utf8').replace(/\r\n/g, '\n');
  const envSource = readFileSync(new URL('../../../../.env.example', import.meta.url), 'utf8').replace(/\r\n/g, '\n');

  assert.match(settingsSource, /key:\s*'PEXELS_API_KEY'/);
  for (const [label, source] of [
    ['settings', settingsSource], ['installation guide', installationSource], ['env example', envSource],
  ]) {
    assert.match(source, /https:\/\/www\.pexels\.com\/api\/key\//, `${label}: current key URL`);
    assert.doesNotMatch(source, /pexels\.com\/api\/new\//, `${label}: obsolete key URL`);
    assert.match(source,
      /An API key alone does not authorize\s+dataset or machine-learning use/,
      `${label}: API key is not permission`);
    assert.match(source, /Pexels\s+has explicitly authorized this use case/,
      `${label}: explicit authorization gate`);
  }
  assert.match(readmeSource, /Pexels requires explicit dataset\/ML authorization/);
  for (const source of [settingsSource, installationSource]) {
    assert.match(source,
      /https:\/\/help\.pexels\.com\/hc\/en-us\/articles\/900005880463-What-are-the-Terms-and-Conditions/);
  }
  assert.match(panelSource, /\['pexels', 'Pexels'\]/);
  assert.match(panelSource, /buildPexelsSearchUrl/);
  assert.match(panelSource,
    /I confirm I have explicit Pexels authorization for dataset\/ML use/);
  assert.match(panelSource,
    /https:\/\/help\.pexels\.com\/hc\/en-us\/articles\/900005880463-What-are-the-Terms-and-Conditions/);
  assert.match(panelSource, /Photos provided by Pexels/);
  assert.match(panelSource, /<PexelsAttribution metadata=\{it\}/);
  // The url→payload mapping lives in scraperSourceSearch.js (scrapeItemToImportPayload),
  // not inlined in the panel — see scraperSourceSearch.test.js for its behavioural coverage.
  // Pinned on the CALL SITE, not just the import: a dangling import with nothing
  // calling it would still match a bare name-presence check, and the wiring this
  // test exists to guarantee (selected items are actually mapped before import)
  // would then be unpinned.
  assert.match(panelSource, /\.map\(scrapeItemToImportPayload\)/);
  for (const field of ['platform', 'source_url', 'photographer', 'photographer_url']) {
    assert.match(scraperSourceSearchSource, new RegExp(`${field}:`), `selected items forward ${field}`);
  }
  assert.match(attributionSource, /Photo by\{' '\}[\s\S]*\{' · '\}[\s\S]*Pexels/);
  assert.match(attributionSource, /rel="noopener noreferrer"/);

  const selectionButton = panelSource.match(
    /<button type="button" onClick=\{\(\) => toggle\(it\.url\)\}[\s\S]*?<\/button>/);
  assert.ok(selectionButton, 'selection button markup must remain present');
  assert.doesNotMatch(selectionButton[0], /<a\b/i,
    'Pexels credit links must remain siblings of the selection button');
});

test('Vast offer filter terms find the Cloud training plugin settings', () => {
  const training = cloud.slots['settings.group'].find(group => group.id === 'cloud');
  for (const query of ['verified', 'secure cloud', 'community cloud', 'offer filter']) {
    assert.equal(matchesQuery(training, query), true, query);
  }
});
