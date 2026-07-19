import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const lightbox = readFileSync(new URL('./DatasetLightbox.jsx', import.meta.url), 'utf8');
const workspace = readFileSync(new URL('./DatasetWorkspace.jsx', import.meta.url), 'utf8');
const hook = readFileSync(new URL('../../hooks/useDataset.js', import.meta.url), 'utf8');
const settings = readFileSync(new URL('../settings/ScrapingSection.jsx', import.meta.url), 'utf8');
const attribution = readFileSync(new URL('./PexelsAttribution.jsx', import.meta.url), 'utf8');

test('lightbox exposes an accessible responsive image improvement action', () => {
  assert.match(lightbox, /workspace\.lightbox\.improve/);
  assert.match(lightbox, /workspace\.lightbox\.improving/);
  assert.match(lightbox, /workspace\.lightbox\.reviewImprovement/);
  assert.match(lightbox, /aria-busy=\{improvementActive\}/);
  assert.match(lightbox, /w-full sm:w-auto/);
  assert.match(lightbox, /workspace\.lightbox\.improveHelp/);
  assert.match(lightbox, /busy \|\| improvementActive \|\| improveReady \|\| !kleinAvailable/);
});

test('workspace guards rescue rows and detects a pending improvement child', () => {
  assert.match(workspace, /!viewImgLive\._rescueReviewPreview/);
  assert.match(workspace, /!isSmallImageRescueRow\(viewImgLive\)/);
  assert.match(workspace, /viewImgLive\.derivation_kind !== 'klein_image_improve'/);
  assert.match(workspace, /image\.derivation_kind === 'klein_image_improve'/);
  assert.match(workspace, /image\.parent_image_id === viewImgLive\.id/);
  assert.match(workspace, /const viewImgImproving[\s\S]*image\.status === 'pending'[\s\S]*\)\) : false/);
  assert.match(workspace, /const viewImgImprovementReady[\s\S]*image\.status === 'pending'[\s\S]*!!image\.filename/);
  assert.match(workspace, /kleinAvailable=\{Boolean\(caps\.engines\?\.klein\)\}/);
});

test('dataset hook starts improvement, reports the preserved original, then refreshes', () => {
  assert.match(hook, /`\/api\/dataset\/image\/\$\{imageId\}\/improve`, \{\}/);
  assert.match(hook, /original stays intact while a separate 2 MP candidate is generated for validation/);
  assert.match(hook, /Could not start image improvement/);
  assert.match(hook, /resolveSmallImageRescue, improveImage, classify/);
});

test('settings separates scraper rescue instructions from manual lightbox improvement', () => {
  assert.match(settings, /settings\.scraping\.improvementTitle/);
  assert.match(settings, /settings\.scraping\.improvementHelp/);
  assert.match(settings, /settings\.scraping\.instructionHelp/);
});

test('manual improvement candidates cannot use the unrelated generic regenerate path', () => {
  const gridItem = readFileSync(new URL('./DatasetGridItem.jsx', import.meta.url), 'utf8');
  assert.match(gridItem, /const isImageImproveCandidate = img\.derivation_kind === 'klein_image_improve'/);
  assert.match(gridItem, /!isRescueDerived && !isImageImproveCandidate && img\.source === 'generated'/);
  assert.match(gridItem, /if \(!isImageImproveCandidate && img\.status !== 'reject'/);
});

test('curation grid and lightbox render the persisted safe Pexels attribution', () => {
  const gridItem = readFileSync(new URL('./DatasetGridItem.jsx', import.meta.url), 'utf8');
  assert.match(gridItem, /<PexelsAttribution metadata=\{img\.source_metadata\}/);
  assert.match(lightbox, /<PexelsAttribution metadata=\{img\.source_metadata\}/);
  assert.match(attribution, /Photo by\{' '\}/);
  assert.match(attribution, /rel="noopener noreferrer"/);
  assert.match(attribution, /attribution\.photographerUrl/);
  assert.match(attribution, /attribution\.sourceUrl/);
});
