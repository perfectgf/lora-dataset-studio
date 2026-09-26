import { registerBundledDescriptor } from '../../../frontend/tests/support/bundledDescriptors.mjs'
import test from 'node:test'
import assert from 'node:assert/strict'
import { existsSync, readFileSync } from 'node:fs'
import { matchesQuery, SETTINGS_SECTIONS } from '../../../frontend/src/components/settings/registry.js'

import descriptor from '../frontend/index.js'
import { contributions, helpTopics } from '../../../frontend/src/plugins/registry.js'
assert.equal(registerBundledDescriptor(descriptor), true)
const engines = readFileSync(new URL('../../../frontend/src/components/settings/EnginesSection.jsx', import.meta.url), 'utf8')
const scraping = readFileSync(new URL('../frontend/panels/ScrapeSettingsGroup.jsx', import.meta.url), 'utf8')

test('small-image rescue search metadata belongs to Web scraping, outside general services', () => {
  const scraping = SETTINGS_SECTIONS.find((section) => section.id === 'scraping');
  const group = contributions('settings.group').find(item => item.id === 'web-scraping');
  assert.equal(group.plugin, 'scrape');
  for (const query of ['klein', 'small image', 'rescue', 'upscale']) {
    assert.equal(matchesQuery(scraping, query), false, `${query}: absent from core services`);
    assert.equal(matchesQuery(group, query), true, `${query}: retained in the product`);
  }
  const topic = helpTopics().find(item => item.id === 'klein.small_image_prompt');
  assert.equal(topic.app.route, '/plugins/scrape/settings');
  assert.equal(topic.app.focus, 'klein-small-image-prompt');
});

test('Pexels terms stay with the scraper while the core shared key remains discoverable', () => {
  const scraping = SETTINGS_SECTIONS.find((section) => section.id === 'scraping');
  const group = contributions('settings.group').find(item => item.id === 'web-scraping');
  for (const query of ['pexels', 'pexels api', 'quota']) {
    assert.equal(matchesQuery(scraping, query), false, query);
    assert.equal(matchesQuery(group, query), true, query);
  }
  assert.equal(matchesQuery(scraping, 'api key'), true, 'shared Civitai access is still core');
  const topic = helpTopics().find(item => item.id === 'PEXELS_API_KEY');
  assert.equal(topic.app.route, '/plugins/scrape/settings');
  assert.equal(topic.app.focus, 'PEXELS_API_KEY');
});

test('Pexels key and attribution markup stay wired without nested controls', () => {
  const panelUrl = new URL('../frontend/panels/ConceptSourcesPanel.jsx', import.meta.url);
  if (!existsSync(panelUrl)) return;   // the scrape plugin is absent: no panel to hold to the section
  const settingsSource = readFileSync(new URL('../frontend/panels/ScrapeSettingsGroup.jsx', import.meta.url), 'utf8');
  const panelSource = readFileSync(panelUrl, 'utf8');
  const scraperSourceSearchSource = readFileSync(
    new URL('../frontend/lib/scraperSourceSearch.js', import.meta.url), 'utf8');
  const attributionSource = readFileSync(
    new URL('../../../frontend/src/components/dataset/PexelsAttribution.jsx', import.meta.url), 'utf8');
  const readmeSource = readFileSync(new URL('../../../README.md', import.meta.url), 'utf8');
  const installationSource = readFileSync(new URL('../../../docs/guide/installation.md', import.meta.url), 'utf8');
  const envSource = readFileSync(new URL('../../../.env.example', import.meta.url), 'utf8');

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
  assert.match(panelSource, /\.filter\(\(it\) => selected\.has\(it\.url\)[\s\S]*?\.map\(scrapeItemToImportPayload\)/);
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

test('scraper rescue distinguishes its instruction from the optional Improve product', () => {
  // The optional scraper owns its rescue explanation. The core card remains
  // usable on its own and no longer advertises a product it does not require.
  assert.doesNotMatch(engines, /Klein rescue — small scraped images/);
  assert.match(engines, /id="identity-prompts"/);
  assert.match(engines, /title="Identity, Klein & Krea 2 prompts \(advanced\)"/);
  // The manual improvement prompt moved with Improve. Sending the user to
  // core identity prompts now names a card that cannot answer this instruction.
  assert.match(scraping, /title="Klein rescue — small scraped images"/);
  assert.match(scraping, /Small-image rescue instruction/);
  assert.match(scraping, /optional Klein Improve plug-in/);
  assert.doesNotMatch(scraping, /Settings ▸ Engines ▸ “Identity, Klein &amp; Krea 2 prompts”/);
});

test('klein.small_image_prompt stays a genuinely optional EMPTY field', () => {
  // It is NOT part of the single-box migration: its config default is '' with no
  // shipped text behind it (backend reads klein.small_image_prompt, '') — empty
  // means "no instruction at all", not "use a built-in one". Pre-filling it would
  // invent a rescue prompt on the user's behalf.
  assert.doesNotMatch(scraping, /PromptOverrideField/);
  assert.match(scraping, /placeholder="Empty — reference image only"/);
});
