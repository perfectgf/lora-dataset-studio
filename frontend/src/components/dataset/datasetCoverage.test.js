import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  axisRows, axisSummary, coverageFilterLabel, coverageReadiness, coverageScope,
  generateMoreHint,
} from './datasetCoverage.js';

const here = path.dirname(fileURLToPath(import.meta.url));

// A payload shaped exactly like the server's: eight front-on studio portraits,
// which is the case the whole feature exists for — the composition bar can call
// that set complete while it has no profile and no three-quarter at all.
const FRONT_ONLY = {
  total: 8,
  captioned: 8,
  uncaptioned: 0,
  kind: 'character',
  axes: [{
    id: 'view',
    label: 'Camera view',
    mode: 'buckets',
    hint: 'Which side of the subject the set ever saw',
    buckets: [
      { id: 'frontal', label: 'frontal', count: 8, core: true, thin: false },
      { id: 'three_quarter', label: 'three-quarter', count: 0, core: true, thin: false },
      { id: 'profile', label: 'profile', count: 0, core: true, thin: false },
      { id: 'from_behind', label: 'from behind', count: 1, core: false, thin: true },
    ],
  }],
  advice: [{ tone: 'warn', text: 'No three-quarter or profile mentioned in any caption.' }],
};

test('axisRows marks a missing CORE bucket as a gap, an optional one as merely absent', () => {
  const byId = Object.fromEntries(axisRows(FRONT_ONLY.axes[0]).map((r) => [r.id, r]));
  assert.equal(byId.frontal.state, 'ok');
  assert.equal(byId.profile.state, 'gap');
  assert.equal(byId.three_quarter.state, 'gap');
  // Seen once in a set of eight: present, but not enough to generalise.
  assert.equal(byId.from_behind.state, 'thin');
});

test('axisSummary says what IS there before what is missing', () => {
  const s = axisSummary(FRONT_ONLY.axes[0]);
  assert.match(s, /frontal 8/);
  assert.match(s, /no three-quarter, no profile/);
  assert.ok(s.indexOf('frontal 8') < s.indexOf('no three-quarter'));
});

test('axisSummary is honest when nothing on the axis was named at all', () => {
  const empty = { label: 'Lighting', buckets: [{ id: 'a', label: 'daylight', count: 0, core: true }] };
  assert.match(axisSummary(empty), /nothing in the captions names one/);
});

test('generateMoreHint names the concrete things to add, and only core gaps', () => {
  const hint = generateMoreHint(FRONT_ONLY);
  assert.match(hint, /Generate or import more/);
  assert.match(hint, /three-quarter/);
  assert.match(hint, /profile/);
  assert.doesNotMatch(hint, /from behind/);   // optional bucket, not a demand
});

test('generateMoreHint stays silent rather than inventing a suggestion', () => {
  const covered = { axes: [{ buckets: [{ id: 'a', label: 'frontal', count: 3, core: true }] }] };
  assert.equal(generateMoreHint(covered), '');
  assert.equal(generateMoreHint(null), '');
});

test('coverageReadiness explains an empty panel instead of showing nothing', () => {
  assert.equal(coverageReadiness(null).ready, false);
  const noCaptions = coverageReadiness({ total: 12, captioned: 0 });
  assert.equal(noCaptions.ready, false);
  assert.match(noCaptions.reason, /No captions yet/);
  assert.match(noCaptions.reason, /caption pass/);
  const noImages = coverageReadiness({ total: 0, captioned: 0 });
  assert.match(noImages.reason, /No images yet/);
  assert.equal(coverageReadiness(FRONT_ONLY).ready, true);
});

test('coverageScope names the pool and the images it could not read', () => {
  assert.equal(coverageScope(FRONT_ONLY), '8 of 8 images captioned');
  assert.match(coverageScope({ total: 10, captioned: 6, uncaptioned: 4 }), /4 not read/);
  // The pool includes undecided images, so it must NOT be called "kept".
  assert.doesNotMatch(coverageScope(FRONT_ONLY), /kept/);
});

// --- contract: the panel must keep saying what it cannot see ----------------
// Read as TEXT (node --test cannot parse JSX). These sentences are the whole
// reason the feature is honest rather than authoritative; a rewrite that drops
// them ships a keyword scan disguised as analysis.
test('the panel keeps its honesty footer and never claims to act', () => {
  const jsx = fs.readFileSync(path.join(here, 'CoveragePanel.jsx'), 'utf8');
  assert.match(jsx, /reads the words in your\s*\n?\s*captions, not the pixels/);
  assert.match(jsx, /Advice only — nothing is kept, rejected or changed/);
  assert.match(jsx, /still counts as a smile/);   // the known false positive
});

test('the panel is wired in under the composition bar', () => {
  const ws = fs.readFileSync(path.join(here, 'DatasetWorkspace.jsx'), 'utf8');
  const bar = ws.indexOf('<CompositionBar');
  const panel = ws.indexOf('<CoveragePanel');
  assert.ok(bar > 0 && panel > bar, 'CoveragePanel must render after CompositionBar');
});

// ---- a chip that knows its images can become a filter ----------------------

// The same axis as above, now with the ids the dataset payload carries.
const WITH_IDS = {
  id: 'view',
  label: 'Camera view',
  mode: 'buckets',
  buckets: [
    { id: 'frontal', label: 'frontal', count: 2, core: true, thin: false, image_ids: [7, 9] },
    { id: 'profile', label: 'profile', count: 0, core: true, thin: false, image_ids: [] },
  ],
};

test('axisRows carries the ids through, and tells "none" apart from "never said"', () => {
  const withIds = Object.fromEntries(axisRows(WITH_IDS).map((r) => [r.id, r]));
  assert.deepEqual(withIds.frontal.imageIds, [7, 9]);
  // Counted nobody, but the payload DID answer: an empty list, not null.
  assert.deepEqual(withIds.profile.imageIds, []);
  // A payload that never mentioned ids at all (the bank's) reads as null, which
  // is what keeps its chips from turning into buttons that filter to nothing.
  const silent = Object.fromEntries(axisRows(FRONT_ONLY.axes[0]).map((r) => [r.id, r]));
  assert.equal(silent.frontal.imageIds, null);
});

test('a coverage pick is named by its axis as well as its bucket', () => {
  // "studio" is a light on one axis and a place on another: the bucket alone
  // would put two different filters under the same name in the filter bar.
  const [frontal] = axisRows(WITH_IDS);
  assert.equal(coverageFilterLabel(WITH_IDS, frontal), 'frontal — camera view');
});

test('the grid narrows to the ids a chip picked, and says so in the filter bar', () => {
  const ws = fs.readFileSync(path.join(here, 'DatasetWorkspace.jsx'), 'utf8');
  // The pick both filters AND jumps to the grid: the panel lives in another
  // section, and a filter applied out of sight reads as nothing happening.
  assert.match(ws, /setCoverageFilter\(pick\); setSection\('images'\)/);
  // It narrows by id — never by re-running the caption lexicon in the browser.
  assert.match(ws, /coverageIds\.has\(i\.id\)/);
  // It is a filter like the others: it shows in the bar, it clears with the rest.
  assert.match(ws, /coverageLabel=\{coverageFilter \? coverageFilter\.label : ''\}/);
  assert.match(ws, /filtersActive = tagFiltersActive \|\| statusFilterActive \|\| !!coverageFilter/);
  const clearAll = ws.slice(ws.indexOf('const clearFilters'), ws.indexOf('const tagFiltersActive'));
  assert.match(clearAll, /setCoverageFilter\(null\)/, 'clear all must clear it too');
});

test('the bank panel says WHY its chips are not buttons', () => {
  // CLAUDE.md: a shared feature ships on both surfaces, or names the reason.
  const bank = fs.readFileSync(
    path.join(here, '..', 'bank', 'BankCoveragePanel.jsx'), 'utf8');
  assert.match(bank, /DELIBERATE DIVERGENCE/);
  assert.match(bank, /pages over SQL/);
});
