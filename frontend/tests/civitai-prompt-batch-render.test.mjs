/**
 * 📝 Le lot Civitai, EXÉCUTÉ — pas lu comme du texte.
 *
 * `civitaiBrowser.contract.test.js` épingle qui passe quoi à qui. C'est utile et
 * c'est aveugle à la seule chose qui casse un écran : le rendu. La modale
 * appelle `useToast()` à sa racine (qui JETTE hors de son provider), porte un
 * `<HelpBadge>` (`useNavigate`) et un `<Link>` ; sa branche `batchable` est du
 * JSX qu'aucun test n'exécutait. Un ReferenceError là-dedans parse, passe le
 * lint, passe le contrat de source, et blanchit l'écran au premier clic.
 *
 * Ce que ce fichier prouve, et rien de plus :
 *   1. la modale rend dans les DEUX états, avec et sans lot ;
 *   2. sans hôte accepteur, elle ne fait pousser AUCUN pied de lot — la moitié
 *      qu'un test de feature oublie toujours ;
 *   3. le pied annonce ce que le lot contient.
 *
 * ⚠️ Ce qu'il NE prouve PAS, dit franchement : aucun événement ne part
 * (`renderToStaticMarkup`), et les cartes arrivent par un `useEffect` inerte
 * côté serveur — la liste est VIDE ici, donc la case d'UNE carte n'est pas
 * exercée. Elle reste couverte par le contrat de source et, désormais, par
 * l'état `civitai` de la sonde responsive.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { createElement, render } from './support/mountJsx.mjs'

const { default: CivitaiBrowserModal } =
  await import('../src/components/dataset/studio/CivitaiBrowserModal.jsx')
const { default: CivitaiBrowserButton } =
  await import('../src/components/dataset/studio/CivitaiBrowserButton.jsx')
/* La modale porte un <HelpBadge> (useNavigate) et un <Link> vers les réglages :
   dans l'app elle vit sous le routeur, le harnais ne le fournit pas. */
const { MemoryRouter } = await import('react-router')

const noop = () => {}
const underRouter = (Component) => (props) =>
  createElement(MemoryRouter, null, createElement(Component, props))

const modal = (props) => render(underRouter(CivitaiBrowserModal),
  { open: true, onClose: noop, onUse: noop, ...props })

test('la modale rend SANS lot — l’état d’avant la feature, intact', () => {
  const html = modal({})
  assert.ok(html.includes('Civitai top prompts'), 'la modale doit rendre')
  assert.ok(!html.includes('civitai-batch-footer'), 'aucun pied de lot sans hôte accepteur')
  assert.ok(!html.includes('role="checkbox"'), 'aucune case sans hôte accepteur')
})

test('la modale rend AVEC un lot — la branche batchable s’exécute vraiment', () => {
  const html = modal({ picks: ['un prompt', 'un autre'], onTogglePick: noop })
  assert.ok(html.includes('civitai-batch-footer'), 'le pied du lot manque')
  assert.match(html, /2 prompts/, 'le pied doit dire ce que le lot contient')
})

test('un lot vide garde la modale cochable, sans pied à zéro', () => {
  const html = modal({ picks: [], onTogglePick: noop })
  assert.ok(html.includes('Civitai top prompts'))
  assert.ok(!html.includes('civitai-batch-footer'), 'pas de pied à zéro')
})

test('le bouton rend dans les deux états', () => {
  const withPicks = render(underRouter(CivitaiBrowserButton),
    { prompt: '', onPrompt: noop, picks: ['a', 'b', 'c'], onTogglePick: noop })
  assert.ok(withPicks.includes('Civitai'))
  const bare = render(underRouter(CivitaiBrowserButton), { prompt: '', onPrompt: noop })
  assert.ok(bare.includes('Civitai'))
})
