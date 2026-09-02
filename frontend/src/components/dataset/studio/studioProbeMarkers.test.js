/* 📐 The markers the responsive probe measures the Test Studio by.
 *
 * Same contract as bankProbeMarkers.test.js: `scripts/responsiveProbe.mjs`
 * finds its surfaces by attribute, and these assertions keep the attributes
 * in place — the probe is the only thing that can SEE the layout, this file
 * only guarantees it is still looking at the right elements.
 *
 * First measured 2026-08-23: 48 violations on the first run (the bottom bar's
 * 27-px shortcut chips and its Generate button, at every width below lg), 0
 * after. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const read = (rel) => fs.readFileSync(new URL(rel, import.meta.url), 'utf8');
const shell = read('./StudioShell.jsx');
const actionBar = read('./StudioActionBar.jsx');
const setup = read('./RunSetupPanel.jsx');
const picker = read('./LoraPicker.jsx');
const probe = read('../../../../scripts/responsiveProbe.mjs');

test('the Studio chrome and panels are marked for the responsive probe', () => {
  assert.match(shell, /<header data-probe-chrome="header"/);
  assert.match(actionBar, /<nav aria-label="Studio quick navigation" data-probe-chrome="action-bar"/);
  assert.match(setup, /data-probe-panel="setup"/);
  assert.match(picker, /data-probe-panel="picker"/);
});

test('the bottom bar is finger-sized below lg, untouched on a desktop', () => {
  // the shortcut chips and the Generate button both carry the pattern
  assert.equal((actionBar.match(/min-h-10 lg:min-h-0/g) || []).length, 2);
});

test('the probe reaches the Studio through its id-carrying route', () => {
  assert.match(probe, /'#\/dataset\/studio': \{/);
  // `#/dataset/studio/1` must land on the Studio spec, not on `#/datasets` and
  // not on nothing: the lookup is longest-prefix over segment boundaries.
  assert.match(probe, /const hashPath = \(\(args\.url\.split\('#'\)\[1\] \|\| '\/'\)\.split\('\?'\)\[0\]\);/);
  assert.match(probe, /return hashPath === p \|\| hashPath\.startsWith\(p \+ '\/'\);/);
  assert.match(probe, /\.sort\(\(a, b\) => b\.length - a\.length\)\[0\] \|\| null;/);
  // an unknown page is still measured at rest — never skipped silently
  assert.match(probe, /const pageSpec = \(route && PAGES\[route\]\) \|\| UNKNOWN_PAGE;/);
});

test('the shortcut state drives the bar the users drive', () => {
  assert.match(probe, /\{ name: 'shortcut', open: \['\[data-probe-chrome="action-bar"\] button'\] \}/);
});

test('the probe opens the VIDEO lane, and the tab whose grid lives deeper', () => {
  /* The lane is a tab: every run before this measured the Images lane and
     called the page clean, while the video panels had never been seen at
     360 px. The selectors are pinned on both sides — a renamed testid would
     otherwise put the lane back out of sight with the probe still green. */
  const lanes = read('../../../pages/StudioPage.jsx');
  const picker = read('./video/VideoSourcePicker.jsx');
  assert.match(lanes, /data-testid=\{`studio-lane-\$\{id\}`\}/);
  assert.match(picker, /data-testid=\{`video-source-\$\{id\}`\}/);
  assert.match(probe, /\{ name: 'video', open: \['\[data-testid="studio-lane-video"\]'\] \}/);
  assert.match(probe, /'\[data-testid="video-source-gallery"\]'/);
  assert.match(probe, /'\[data-testid="video-source-clip"\]'/);
});

test('the probe opens the 🌐 Civitai browser, whose action row grew a third button', () => {
  /* Le lot de prompts a fait passer la rangée d'actions d'une carte de deux
     boutons à trois, dans une colonne qui fait ~250 px à 360 px de large à côté
     de la vignette. Aucun état de sonde n'ouvrait cette modale : la rangée
     n'avait donc jamais été mesurée à aucune taille. Les deux côtés sont
     épinglés — un bouton renommé remettrait la surface hors de portée avec la
     sonde toujours verte, ce qui est exactement le trou que ce fichier existe
     pour fermer. */
  const modal = read('./CivitaiBrowserModal.jsx');
  const button = read('./CivitaiBrowserButton.jsx');
  // La modale couvre la page par design : layer (non budgétée), et un panneau
  // nommé pour que la mesure de remplissage la voie.
  assert.match(modal, /data-probe-layer data-probe-panel="civitai-browser"/);
  // Le texte du bouton EST le sélecteur de la sonde.
  assert.match(button, /🌐 Civitai/);
  assert.match(probe, /\{ name: 'civitai', open: \['button:has-text\("🌐 Civitai"\)'\] \}/);
});
