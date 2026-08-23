import assert from 'node:assert/strict'
import test from 'node:test'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import {
  PASSES_PANEL_DEFAULT_OPEN, RAIL_FOLDED_GROUPS, RAIL_PRIMARY_GROUPS,
  RAIL_SIDE_BY_SIDE_PX, RAIL_STORAGE_KEY,
  foldedCount, loadRailOpen, passesButtonLabel, passesPanelStartsOpen,
  railDefaultOpen, railGroups, railIsColumn, saveRailOpen,
} from './bankLayout.js'

/** A localStorage stand-in. `failing` reproduces the private-mode browser, which
 *  throws on ACCESS — the case that made gridSort.js inject its store too. */
function fakeStore(initial = {}, { failing = false } = {}) {
  const data = new Map(Object.entries(initial))
  return {
    getItem(k) { if (failing) throw new Error('denied'); return data.has(k) ? data.get(k) : null },
    setItem(k, v) { if (failing) throw new Error('denied'); data.set(k, String(v)) },
    removeItem(k) { if (failing) throw new Error('denied'); data.delete(k) },
    get size() { return data.size },
  }
}

test('the rail sits beside the grid only once the grid still has room', () => {
  assert.equal(railIsColumn(1440), true)
  assert.equal(railIsColumn(RAIL_SIDE_BY_SIDE_PX), true)
})

/* The threshold is about the GRID, not about the rail. A 17rem rail fits from
   ~640 px, which is why it first sat there — but it left the grid ~350 px, two
   thumbnails wide, and the screen stopped being a triage screen. The rail is a
   drawer until the grid keeps a workable width. */
test('a 17rem rail never leaves the grid narrower than a usable strip', () => {
  const RAIL_PX = 17 * 16
  assert.ok(RAIL_SIDE_BY_SIDE_PX - RAIL_PX >= 700,
    `at ${RAIL_SIDE_BY_SIDE_PX}px the grid would get ${RAIL_SIDE_BY_SIDE_PX - RAIL_PX}px`)
  assert.equal(railIsColumn(800), false, '800px still belongs to the drawer')
})

test('at 400 px the rail is a drawer, not a column — it folds instead of overflowing', () => {
  // The whole point of the 400 px requirement: a 15-rem rail beside a grid at
  // this width leaves neither usable.
  assert.equal(railIsColumn(400), false)
  assert.equal(railDefaultOpen(400), false)
})

test('an unknown width answers "beside", never "hidden"', () => {
  // Guessing "drawer" on a real desktop would hide the filters behind a click
  // nobody was told about. The desktop layout is the safe degradation.
  for (const w of [undefined, null, NaN, 'wide']) {
    assert.equal(railIsColumn(w), true, `width ${String(w)}`)
    assert.equal(railDefaultOpen(w), true, `width ${String(w)}`)
  }
})

test('the rail opens by default on a desktop', () => {
  assert.equal(railDefaultOpen(1440), true)
})

test('a remembered rail state wins over the width default', () => {
  assert.equal(loadRailOpen(1440, fakeStore({ [RAIL_STORAGE_KEY]: 'closed' })), false)
  assert.equal(loadRailOpen(400, fakeStore({ [RAIL_STORAGE_KEY]: 'open' })), true)
})

test('a corrupt preference degrades to the width default, never to "no filters"', () => {
  // A stored value from a future build, a truncated write, anything: the user
  // must not end up on a screen with no way to reach the filters.
  for (const raw of ['', 'yes', '1', 'OPEN', '{}']) {
    assert.equal(loadRailOpen(1440, fakeStore({ [RAIL_STORAGE_KEY]: raw })), true, raw)
    assert.equal(loadRailOpen(400, fakeStore({ [RAIL_STORAGE_KEY]: raw })), false, raw)
  }
})

test('a storage that throws on access still yields a usable rail', () => {
  assert.equal(loadRailOpen(1440, fakeStore({}, { failing: true })), true)
  assert.equal(loadRailOpen(400, fakeStore({}, { failing: true })), false)
  // …and saving must not throw either: a full quota must not break a toggle.
  assert.equal(saveRailOpen(true, fakeStore({}, { failing: true })), true)
})

test('saving then loading round-trips both states', () => {
  const store = fakeStore()
  saveRailOpen(false, store)
  assert.equal(loadRailOpen(1440, store), false)
  saveRailOpen(true, store)
  assert.equal(loadRailOpen(400, store), true)
})

test('the everyday facets stay visible and the measured axes fold below', () => {
  const { primary, folded } = railGroups()
  assert.deepEqual(primary, ['status', 'quality', 'groups'])
  assert.deepEqual(folded, ['score', 'framing', 'medium', 'angle', 'resolution', 'origin'])
})

test('a group whose pass produced nothing is omitted, never rendered empty', () => {
  // An empty "🎨 Medium" heading reads as a broken pass rather than an unrun one.
  const { primary, folded } = railGroups({ medium: false, origin: false, quality: false })
  assert.deepEqual(primary, ['status', 'groups'])
  assert.deepEqual(folded, ['score', 'framing', 'angle', 'resolution'])
})

test('the folded count is what the disclosure prints, and it follows availability', () => {
  assert.equal(foldedCount(), RAIL_FOLDED_GROUPS.length)
  assert.equal(foldedCount({ score: false, angle: false }), RAIL_FOLDED_GROUPS.length - 2)
})

test('no facet is claimed by both halves of the rail', () => {
  const overlap = RAIL_PRIMARY_GROUPS.filter((g) => RAIL_FOLDED_GROUPS.includes(g))
  assert.deepEqual(overlap, [])
})

test('the passes button says the bank is busy, because the panel can be closed over a run', () => {
  assert.equal(passesButtonLabel(false), '⚙ Passes')
  assert.equal(passesButtonLabel(true), '⚙ Passes (running…)')
})

test('the passes panel starts closed on a bank that has already been scanned', () => {
  assert.equal(passesPanelStartsOpen({ total: 9000, scanned: 9000 }), false)
  assert.equal(passesPanelStartsOpen({ total: 9000, scanned: 1 }), false)
})

test('a freshly imported dump opens the panel — the grid alone offers nothing to do', () => {
  assert.equal(passesPanelStartsOpen({ total: 9000, scanned: 0 }), true)
})

test('an unloaded payload does not pop the panel open under the user', () => {
  assert.equal(passesPanelStartsOpen(null), PASSES_PANEL_DEFAULT_OPEN)
  assert.equal(passesPanelStartsOpen(undefined), PASSES_PANEL_DEFAULT_OPEN)
  assert.equal(passesPanelStartsOpen({}), PASSES_PANEL_DEFAULT_OPEN)
  // An empty bank has nothing to scan either — the passes cannot help it.
  assert.equal(passesPanelStartsOpen({ total: 0, scanned: 0 }), false)
})

/* ── The drawer is painted, not tinted ──────────────────────────────────────
   Same trap the Canvas already fell into (canvasChromeLayering.test.js):
   `bg-surface` and `bg-surface-overlay` read as synonyms and are not. The
   first is a 4 %-alpha white tint for lifting a card off the OPAQUE page
   background; the second is the near-opaque panel token for anything that
   FLOATS over content. Painted with the tint, the drawer is a sheet of glass —
   the grid and the header are legible straight through the open filters.
   A source test, because `node --test` cannot parse JSX and the symptom reads
   as a z-index bug nobody can locate. */
test('the filter drawer floats on an OPAQUE panel, never on the card tint', () => {
  const HERE = dirname(fileURLToPath(import.meta.url))
  const src = readFileSync(join(HERE, 'BankFilterRail.jsx'), 'utf8')
  // The drawer branch of the rail's root <aside>: the line that carries the
  // `fixed` positioning is the one that floats over the grid.
  const drawer = src.split('\n').find((l) => /\bfixed\b/.test(l) && /inset-y-0/.test(l))
  assert.ok(drawer, 'the drawer branch must keep its fixed inset-y-0 handle')
  assert.match(drawer, /\bbg-surface-overlay\b/, 'the drawer floats on a transparent panel')
  // …and never the bare tint, nor the opaque token thinned back into glass.
  assert.doesNotMatch(drawer, /\bbg-surface\b(?!-)/)
  assert.doesNotMatch(drawer, /bg-surface-overlay\/\d/)
})

/* The panel opens itself only where it has room to. Measured by the responsive
   probe on a fresh bank at 360 px: ~1 550 px of panel — the eight pass buttons,
   the semantic-engine card, the watermark and edit panels and the overview, all
   stacked — put the first image two screens down. The ⚙ Passes button is in the
   header on every width; below lg it stays a button. */
test('the passes panel does not open itself on a phone, and still does on a desktop', () => {
  const fresh = { total: 9000, scanned: 0 }
  assert.equal(passesPanelStartsOpen(fresh, 360), false)
  assert.equal(passesPanelStartsOpen(fresh, 844), false)
  assert.equal(passesPanelStartsOpen(fresh, 1023), false)
  assert.equal(passesPanelStartsOpen(fresh, 1024), true)
  assert.equal(passesPanelStartsOpen(fresh, 1440), true)
  // no width given (server, tests, old callers) = the historical answer
  assert.equal(passesPanelStartsOpen(fresh), true)
  assert.equal(passesPanelStartsOpen(fresh, undefined), true)
})
