/**
 * Le panneau « Datasets » du ◉ LoRA Canvas n'existe plus : c'est une BARRE.
 *
 * Historique, parce qu'il explique ce que ce fichier garde encore. Le panneau
 * s'ouvrait déplié sur écran large ; sa liste de cases (une par dataset, plus
 * les modèles, les statuts, la recherche) poussait le board — ce qu'on vient
 * regarder — sous la ligne de flottaison. Le premier correctif l'a fait ouvrir
 * REPLIÉ. Mesuré ensuite sur une vraie bibliothèque de quatorze datasets en
 * 1280×720, déplié il faisait toujours 389 px, soit 54 % de l'écran, pour
 * quiconque l'avait laissé ouvert une fois — un pli qu'il faut refaire avant de
 * pouvoir travailler n'est pas une réponse. Il a donc été remplacé le
 * 08/08/2026 par une rangée de pastilles d'environ 40 px dont les contrôles
 * vivent dans des popovers.
 *
 * Ce qui est épinglé ici : les helpers `lds.canvasFilterOpen` restent EXPORTÉS
 * et corrects. Ils ne sont plus lus par le composant (une barre n'a pas de pli),
 * mais la clé est écrite dans de vrais navigateurs et la règle du repo est
 * qu'on ne renomme ni ne supprime un identifiant stocké sans chemin d'alias.
 * Le jour où quelque chose voudra à nouveau se souvenir d'un pli sur ce board,
 * il trouvera la clé intacte plutôt que d'en inventer une seconde.
 */
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

import {
  CANVAS_FILTER_OPEN_KEY, readCanvasFilterOpen, writeCanvasFilterOpen,
} from '../src/utils/canvasFamilyFilter.js'

const SOURCE = readFileSync(
  new URL('../src/components/canvas/CanvasDatasetFilter.jsx', import.meta.url), 'utf8')

/** Un localStorage de test, injectable comme le reste des helpers du canvas. */
const store = (initial = {}) => {
  const map = new Map(Object.entries(initial))
  return {
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => map.set(k, String(v)),
    dump: () => Object.fromEntries(map),
  }
}

test('the stored fold key keeps its name and its meaning', () => {
  assert.equal(CANVAS_FILTER_OPEN_KEY, 'lds.canvasFilterOpen')
  assert.equal(readCanvasFilterOpen(store()), false)
})

test('an explicit unfold survives the reload, an explicit fold too', () => {
  const s = store()
  writeCanvasFilterOpen(s, true)
  assert.equal(s.dump()[CANVAS_FILTER_OPEN_KEY], '1')
  assert.equal(readCanvasFilterOpen(s), true)
  writeCanvasFilterOpen(s, false)
  assert.equal(readCanvasFilterOpen(s), false)
})

test('a junk or absent store never throws and reads as folded', () => {
  assert.equal(readCanvasFilterOpen(null), false)
  assert.equal(readCanvasFilterOpen(undefined), false)
  assert.equal(readCanvasFilterOpen(store({ [CANVAS_FILTER_OPEN_KEY]: 'yes' })), false)
  const hostile = { getItem() { throw new Error('private mode') } }
  assert.equal(readCanvasFilterOpen(hostile), false)
  // …et écrire dans un store qui refuse ne casse pas le clic.
  assert.equal(writeCanvasFilterOpen({ setItem() { throw new Error('quota') } }, true), false)
})

test('the filter is a bar: no fold, and no viewport-width logic either', () => {
  // Le pli n'est plus consulté — il n'y a plus de corps dépliable à cacher.
  assert.doesNotMatch(SOURCE, /readCanvasFilterOpen\(/)
  assert.doesNotMatch(SOURCE, /writeCanvasFilterOpen\(/)
  // La largeur d'écran n'a jamais eu le droit de décider de cet état, et ne
  // l'a toujours pas : un agrandissement de fenêtre rouvrait le panneau.
  assert.doesNotMatch(SOURCE, /matchMedia/)
  assert.doesNotMatch(SOURCE, /min-width: 640px/)
  // Chaque contrôle vit dans une pastille à menu, pas dans un corps déplié.
  assert.match(SOURCE, /<CanvasFilterMenu label="Datasets"/)
  assert.match(SOURCE, /<CanvasFilterMenu label="Models"/)
  assert.match(SOURCE, /<CanvasFilterMenu label="Status"/)
})
