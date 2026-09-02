/**
 * Le bouton 🌐 Civitai, EXÉCUTÉ — et pourquoi sa modale, elle, ne peut plus l'être.
 *
 * ── Ce que ce fichier couvre ────────────────────────────────────────────────
 * `CivitaiBrowserButton` rend dans ses deux états (avec et sans lot). C'est un
 * composant ordinaire : le harnais l'exécute, donc un ReferenceError dans une de
 * ses branches devient un test rouge au lieu d'un écran blanc.
 *
 * ── Ce qu'il NE couvre PLUS, et c'est un coût assumé ────────────────────────
 * `CivitaiBrowserModal` est désormais PORTAILLÉE sur `document.body` — le seul
 * fix possible au bug d'empilement (cf. studioModalsArePortaled.contract.
 * test.js : montée sous l'`<aside lg:sticky lg:overflow-auto>`, son z-index est
 * plafonné et sa boîte découpée, et les 👍/👎 des cellules se peignaient
 * par-dessus les prompts).
 *
 * Or `react-dom/server` REFUSE les portails, littéralement — mesuré ici :
 *   « Portals are not currently supported by the server renderer. »
 * Ce n'est donc pas « le markup est vide » : le rendu JETTE. Aucune assertion
 * SSR n'est possible sur cette modale, ni sur son markup ni même sur le simple
 * fait qu'elle s'exécute. Le correctif juste a coûté cette couverture.
 *
 * Ce qui la couvre à la place, et il faut que ce soit dit ici pour que personne
 * ne « répare » le trou en retirant le portail :
 *   · le contrat de source (`civitaiBrowser.contract.test.js`) — qui passe quoi,
 *     et le gate `batchable` ;
 *   · l'état `civitai` de la sonde responsive, qui ouvre la vraie modale dans un
 *     vrai navigateur, à cinq tailles ;
 *   · ⚠️ avec une réserve : `data-probe-layer` est « apparié avec rien » dans le
 *     contrôle de chevauchement de la sonde. Son vert ne dit donc RIEN de ce
 *     bug-là. Sur cette classe de défaut, seule une capture tranche.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { createElement, render } from './support/mountJsx.mjs'

const { default: CivitaiBrowserButton } =
  await import('../src/components/dataset/studio/CivitaiBrowserButton.jsx')
/* Le bouton ouvre une modale qui contient un <Link> : dans l'app il vit sous le
   routeur, le harnais ne le fournit pas. */
const { MemoryRouter } = await import('react-router')

const noop = () => {}
const underRouter = (Component) => (props) =>
  createElement(MemoryRouter, null, createElement(Component, props))

test('le bouton rend sans lot — l’état d’avant la feature', () => {
  const html = render(underRouter(CivitaiBrowserButton), { prompt: '', onPrompt: noop })
  assert.ok(html.includes('Civitai'), 'le bouton doit rendre')
})

test('le bouton rend avec un lot — la branche du compte s’exécute', () => {
  const html = render(underRouter(CivitaiBrowserButton),
    { prompt: '', onPrompt: noop, picks: ['a', 'b', 'c'], onTogglePick: noop })
  assert.ok(html.includes('Civitai'))
})

test('la modale portaillée est HORS de portée du harnais — mesuré, pas supposé', async () => {
  /* La perte de couverture décrite en tête est ÉTABLIE ici, pas affirmée. Le
     jour où react-dom/server saura rendre un portail, ce test rougira — et
     quelqu'un lira l'en-tête et rétablira les assertions perdues au lieu de
     découvrir le trou des années plus tard. */
  const { default: CivitaiBrowserModal } =
    await import('../src/components/dataset/studio/CivitaiBrowserModal.jsx')
  // Sans DOM, on n'atteint même pas createPortal ; on stube le strict minimum
  // pour que le refus du rendu serveur soit ce qu'on observe.
  globalThis.document = { body: { nodeType: 1 } }
  try {
    assert.throws(
      () => render(underRouter(CivitaiBrowserModal),
        { open: true, onClose: noop, onUse: noop, picks: ['a'], onTogglePick: noop }),
      /Portals are not currently supported by the server renderer/,
    )
  } finally {
    delete globalThis.document
  }
})
