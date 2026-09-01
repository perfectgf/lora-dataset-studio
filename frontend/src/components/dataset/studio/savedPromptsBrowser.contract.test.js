import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { THUMB_SIDES } from '../../../utils/datasetThumbUrl.js';

const strip = readFileSync(new URL('./RecentPrompts.jsx', import.meta.url), 'utf8');
const modal = readFileSync(new URL('./SavedPromptsModal.jsx', import.meta.url), 'utf8');
const civitai = readFileSync(new URL('./CivitaiBrowserModal.jsx', import.meta.url), 'utf8');
const promptField = readFileSync(new URL('./PromptField.jsx', import.meta.url), 'utf8');
const canvasSetup = readFileSync(new URL('./StudioRunSetup.jsx', import.meta.url), 'utf8');
const legacyStudio = readFileSync(new URL('./LegacyDatasetStudio.jsx', import.meta.url), 'utf8');
const topics = readFileSync(new URL('../../../help/topics/pages.js', import.meta.url), 'utf8');

/** La valeur d'un `const NOM = <nombre>;` dans une source. */
const constant = (source, name) => {
  const m = new RegExp(`const ${name} = (\\d+);`).exec(source);
  assert.ok(m, `${name} must stay a named constant — a magic number is not testable`);
  return Number(m[1]);
};

test('both launch surfaces reach the browser through the same component', () => {
  // Parité des surfaces de génération : le Studio de test du dataset et
  // « Generate from the board » montent tous deux RecentPrompts, et c'est
  // RecentPrompts qui monte la fenêtre. Ajouter le navigateur d'un seul côté est
  // exactement la divergence silencieuse que les gens remontent comme un bug.
  assert.match(promptField, /<RecentPrompts items=\{recentPrompts\}/);
  assert.match(canvasSetup, /<RecentPrompts items=\{recentPrompts\}/);
  assert.match(strip, /import SavedPromptsModal from '\.\/SavedPromptsModal'/);
  assert.match(strip, /<SavedPromptsModal\b/);
});

test('the browser gets the WHOLE history, the strip only its head', () => {
  // « Browse all 167 » doit ouvrir 167 entrées. Passer la tranche affichée à la
  // fenêtre ferait un bouton qui ment sur son propre compte.
  assert.match(strip, /<SavedPromptsModal[\s\S]*?items=\{items\}/);
  assert.match(strip, /items\.slice\(0, INLINE\)/);
  assert.ok(constant(strip, 'INLINE') > 0 && constant(strip, 'INLINE') <= 12,
    'the strip is a handful, not a wall — that is the whole point of the split');
});

test('the picture is drawn at a size a person can recognise', () => {
  // La panne d'origine : la vignette était en `w-8 h-10` (32×40 px) alors que
  // l'image est le SEUL signal qui distingue deux prompts de 500 caractères qui
  // commencent pareil. Ce test est là pour que personne ne la rétrécisse à
  // nouveau sans s'en rendre compte.
  assert.doesNotMatch(strip, /className="w-8 h-10/,
    'the 32x40 thumbnail is the bug this browser exists to fix');
  assert.match(strip, /h-32 w-24 object-cover/);
  // La fenêtre fait le MÊME geste que le navigateur 🌐 Civitai — choisir un
  // prompt en regardant son image — donc elle le fait à la même taille.
  const rung = 'w-28 sm:w-36 h-40 sm:h-48';
  assert.ok(civitai.includes(rung), 'the Civitai browser is the published reference size');
  assert.ok(modal.includes(rung), 'same job as the Civitai browser, same picture size');
});

test('both thumbnail rungs are ones the server actually materialises', () => {
  // `dataset_thumbs.THUMB_SIDES` : demander un barreau absent ne rend pas une
  // image plus nette, il fait resservir un autre barreau côté serveur.
  for (const [name, source] of [['strip', strip], ['modal', modal]]) {
    const side = constant(source, 'THUMB_SIDE');
    assert.ok(THUMB_SIDES.includes(side), `${name}: ${side} is not a served thumbnail rung`);
    assert.ok(side >= 192, `${name}: ${side} is below what a 96 px tile needs on a 2x screen`);
  }
});

test('every verb the strip offers has a destination in the browser', () => {
  // Porter une feature sur une seconde surface, c'est donner une destination à
  // CHAQUE verbe : recharger, cocher pour le lot, supprimer. Un verbe qui n'existe
  // que dans la bande devient injoignable dès le 7e prompt.
  for (const verb of [/onPick\(/, /onToggleBatch\(/, /onDelete\(/]) {
    assert.match(strip, verb);
    assert.match(modal, verb);
  }
  // …et le lot n'est proposé, des deux côtés, que si l'hôte le passe : « Generate
  // from the board » ne le passe pas, la case ne doit pas y apparaître.
  for (const source of [strip, modal]) {
    assert.match(source, /const batchable = typeof onToggleBatch === 'function';/);
  }
});

test('ticking for the batch never writes into the prompt field', () => {
  // Le contrat du lot : cocher DÉCRIT ce que le prochain lancement rejoue, seul
  // « ⤵ Use prompt » (ou le clic sur une carte) remplit le champ.
  assert.match(modal, /onClick=\{\(\) => onToggleBatch\(p\.prompt\)\}/);
  assert.match(modal, /const use = \(p\) => \{ onPick\(p\); onClose\(\); \};/);
});

test('the browser can be searched and says what it is showing', () => {
  // À ~170 entrées, chercher est le seul moyen de retrouver un prompt — et c'est
  // la forme (`type="search"`) que la Bank, le Canvas et Caption Lab emploient.
  assert.match(modal, /type="search"/);
  assert.match(modal, /filterSavedPrompts\(items, query\)/);
  assert.match(modal, /\$\{shown\.length\} of \$\{total\}/);
  // Un filtre qui ne rend rien doit le DIRE, sinon la fenêtre a l'air cassée.
  assert.match(modal, /No saved prompt contains every word of/);
});

test('the browser is a real dialog', () => {
  assert.match(modal, /role="dialog" aria-modal="true"/);
  assert.match(modal, /useFocusTrap\(ref, open\)/);
  assert.match(modal, /e\.key === 'Escape'/);
  // Clic sur le fond = fermer, mais UNIQUEMENT sur le fond (pas sur une carte).
  assert.match(modal, /if \(e\.target === e\.currentTarget\) onClose\(\)/);
});

test('the browser is portalled out of the launch panel, and the reason still holds', () => {
  // Mesuré en navigateur : sans portail, l'en-tête de l'app et des morceaux de
  // la page se peignaient PAR-DESSUS la fenêtre, et au-delà de `lg` le scroll de
  // l'aside la découpait. Cause : `position: sticky` ouvre un contexte
  // d'empilement, donc aucun z-index posé dedans ne peut en sortir.
  assert.match(modal, /createPortal\(<SavedPromptsPanel \{\.\.\.props\} \/>, document\.body\)/);
  // La prémisse. Si l'aside cesse d'être sticky/scrollable, ce portail se
  // rediscute — il ne se supprime pas en passant.
  assert.match(legacyStudio, /<aside className="[^"]*lg:sticky[^"]*lg:overflow-auto/,
    'the launch panel still lives in a sticky, scrollable aside');
  // Marqué pour la sonde responsive : un dialogue non marqué n'est pas mesuré.
  assert.match(modal, /data-probe-chrome="saved-prompts" data-probe-layer/);
});

test('the help topic the browser badges actually exists', () => {
  assert.match(modal, /<HelpBadge topic="studio-saved-prompts"/);
  assert.match(topics, /action\('studio-saved-prompts',/);
  // Ce que quelqu'un tape quand il vit le symptôme d'origine.
  for (const kw of ['search prompts', 'find a prompt', 'preview too small', 'browse all prompts']) {
    assert.ok(topics.includes(`'${kw}'`), `help keywords must carry “${kw}”`);
  }
});
