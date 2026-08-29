import test from 'node:test'
import assert from 'node:assert/strict'
import { readSource as read } from './support/readSource.mjs'

import { getHelpTopic } from '../src/help/helpRegistry.js'
import { getWorkspacePanel } from '../src/components/dataset/workspaceNavigation.js'

/* WHY THIS TEST EXISTS
 * --------------------
 * The 🧪 Caption Lab shipped reachable from exactly ONE place: the Images grid →
 * a kept tile → its ⤢ button → a tab inside the caption editor. The Captions
 * section — the screen whose entire subject is captions — had no way in, while
 * the app's own help advertised 'caption lab', 'joycaption', 'vocabulary' and
 * 'a/b' on the CAPTIONS SECTION topic and routed all of them to
 * /datasets?section=captions. Searching the app for the bench landed the user on
 * the one screen that did not carry it.
 *
 * Nothing could fail when that happened: a discoverability gap is invisible to
 * logic tests, and the help contract only checks that a route RESOLVES, never
 * that the screen behind it carries what the keywords promised. So this pins the
 * chain end to end — topic → panel → button → dialog opened on the bench — and
 * fails loudly if a later rewrite drops any link of it. */

const bank = read('src/components/bank/BankWorkspace.jsx')
const surfaces = read('src/components/dataset/captionLabSurface.js')
const lab = read('src/components/dataset/CaptionLab.jsx')
const workspace = read('src/components/dataset/DatasetWorkspace.jsx')
const dialog = read('src/components/dataset/CaptionEditorDialog.jsx')
const picker = read('src/components/dataset/CaptionLabPicker.jsx')

test('the Captions section carries a Caption Lab entry point', () => {
  // The row the section rail scrolls to, and the button inside it.
  assert.match(workspace, /id="ds-captions-lab"/,
    'the Captions section lost its Caption Lab row')
  assert.match(workspace, /aria-label="Open the Caption Lab"/,
    'the Caption Lab button lost the label the probe and the user both find it by')
  // …and it renders the picker + the dialog rather than pointing elsewhere.
  assert.match(workspace, /<CaptionLabPicker\s/)
  assert.match(workspace, /<CaptionEditorDialog initialMode="lab"/)
})

test('the rail lists the bench, and the help topic routes to that panel', () => {
  const panel = getWorkspacePanel('captions', 'lab')
  assert.ok(panel, 'the captions section lost its "lab" panel')
  assert.equal(panel.targetId, 'ds-captions-lab')
  assert.equal(panel.title, 'Caption Lab')
  // Gated on KEPT images, not captioned ones: the bench generates, it does not
  // compare stored text — offering it only after a pass would be backwards.
  assert.equal(panel.when, 'hasKeptImages')

  const topic = getHelpTopic('action-caption-lab')
  assert.ok(topic, 'the Caption Lab lost its own help topic')
  assert.equal(topic.app.route, '/datasets?section=captions&panel=lab')
  for (const keyword of ['caption lab', 'compare', 'joycaption', 'vocabulary']) {
    assert.ok(topic.keywords.includes(keyword), `the topic dropped the "${keyword}" keyword`)
  }
})

test('opened from the section, the dialog lands ON the bench', () => {
  /* `initialMode` is read by the useState initialiser, so `labAvailable` must be
     declared ABOVE it — a const declared under a useState initialiser reads
     undefined on the first render, and the dialog would silently open on the
     textarea instead of the bench. Order is the contract here, not just presence. */
  const labAvailableAt = dialog.indexOf('const labAvailable =')
  const modeStateAt = dialog.indexOf('const [mode, setMode] = useState(')
  assert.ok(labAvailableAt > -1 && modeStateAt > -1)
  assert.ok(labAvailableAt < modeStateAt,
    'labAvailable must be declared before the mode initialiser that reads it')
  assert.match(dialog, /useState\(labAvailable && initialMode === 'lab' \? 'lab' : 'edit'\)/)
  // The way back to the picker: benching is a comparison ACROSS rows.
  assert.match(dialog, /onPickAnotherImage/)
  assert.match(workspace, /onPickAnotherImage=\{\(\) => \{ setLabImage\(null\); setLabPickerOpen\(true\); \}\}/)
})

test('both dialogs are measurable surfaces, and opaque cards', () => {
  /* The probe scans touch targets and truncation ONLY inside a
     [data-probe-chrome] subtree, so an unmarked dialog is not "clean" — it is
     unmeasured, which reads exactly the same in a green report. */
  assert.match(dialog, /data-probe-chrome="caption-editor" data-probe-layer/)
  assert.match(picker, /data-probe-chrome="caption-lab-picker" data-probe-layer/)
  // modal-opacity-contract's rule, asserted here too because this picker is a
  // panel over a dim overlay: bg-surface is a 4 % tint, never a card.
  assert.match(picker, /\bbg-app\b/)
})

test('the probe opens the Captions section and walks into the bench', () => {
  const probe = read('scripts/responsiveProbe.mjs')
  for (const state of ['captions', 'caption-lab-picker', 'caption-lab']) {
    assert.match(probe, new RegExp(`\\{ name: '${state}', open:`),
      `the probe lost its '${state}' state — the section stops being measured`)
  }
  // The two selectors the walk hangs on. `:visible` is load-bearing on the rail:
  // the phone's hidden twin nav comes FIRST in the DOM and would swallow the
  // whole page's coverage (see bankProbeMarkers).
  assert.match(probe, /nav\[aria-label="Dataset sections"\]:visible >> button:has-text\("Captions"\)/)
  assert.match(probe, /\[aria-label\^="Bench captions on image"\]:visible/)
  assert.match(picker, /aria-label=\{`Bench captions on image \$\{nameOf\(img\)\}`\}/)
})

// --- the mirror: same bench, on the Bank ---------------------------------------

test('the Bank caption window carries the same entry point', () => {
  assert.match(bank, /aria-label="Open the Caption Lab"/,
    'the Bank lost its Caption Lab button')
  assert.match(bank, /<CaptionLabPicker\s/)
  assert.match(bank, /<CaptionEditorDialog initialMode="lab"/)
  assert.match(bank, /labSurface=\{bankLabSurface\(\{/)
  // The pile is what you are LOOKING at, not "every non-rejected row": a bank pages
  // over SQL and can hold six figures of them. And a selection SURVIVES paging, so an
  // empty intersection falls back to the page rather than greying the button out with
  // a reason ("nothing on this page") that is false while rows sit on screen.
  assert.match(bank, /const scoped = rows\.filter\(\(im\) => selected\.has\(im\.id\)\)/)
  assert.match(bank, /return selected\.size && scoped\.length \? scoped : rows/)
  // …and the write ✓ Keep this one needs, which the Bank had no route for at all.
  assert.match(bank, /\{ caption: nextCaption \}/,
    'the Bank lost the per-image caption write that Keep this one needs')

  const topic = getHelpTopic('bank-caption-lab')
  assert.ok(topic, 'the Bank bench lost its help topic')
  assert.equal(topic.app.route, '/bank')
})

test('the bench itself knows nothing about datasets', () => {
  /* THE PIN THAT MATTERS. CaptionLab used to build `/api/dataset/${datasetId}/…`
     inline, three times. The moment one of those comes back, the Bank silently
     benches against a dataset id — a 404 at best, another bank's row at worst —
     and no test would fail. So the bench may not name an endpoint at all: every
     one of them lives in captionLabSurface.js. */
  assert.doesNotMatch(lab, /\/api\/(dataset|bank)\//,
    'CaptionLab hardcoded a per-surface endpoint again — those belong to the surface')
  // …and no user-visible sentence may name one product either: the blurb used to end
  // with "stores the config for the dataset", which is simply false on a bank.
  assert.doesNotMatch(lab, /for the dataset|dataset's caption options\./,
    'the bench tells the user it is on a dataset — it no longer knows that')
  assert.match(lab, /function CaptionLab\(\{ surface,/)
  for (const call of ['surface.preview(', 'surface.cancel()', 'surface.applyConfig(']) {
    assert.ok(lab.includes(call), `the bench stopped going through ${call}`)
  }
})

test('the two surfaces differ where the products differ, and say so', () => {
  assert.match(surfaces, /export function datasetLabSurface/)
  assert.match(surfaces, /export function bankLabSurface/)
  // A dataset OWNS a caption method; a bank picks one per run. Same promise, two
  // mechanics — so the labels MUST differ. CLAUDE.md: different behaviour must not
  // wear the same label. Equal labels here would be the bug, not the parity.
  const labels = [...surfaces.matchAll(/applyLabel: '([^']+)'/g)].map((m) => m[1])
  assert.equal(labels.length, 2, 'expected exactly one apply label per surface')
  assert.notEqual(labels[0], labels[1],
    'the Bank stores nothing — its button must not promise "make default"')
  // The dataset persists; the bank hands the config to the dials it runs from.
  assert.match(surfaces, /\/caption\/options`, config\)/)
  assert.match(surfaces, /onApplyRunConfig\(config\)/)
})

test('the probe walks into the bench on BOTH surfaces', () => {
  const probe = read('scripts/responsiveProbe.mjs')
  // Datasets states are asserted above; these are the Bank's twins. Without them
  // the ported bench would be measured on one surface and unmeasured on the other,
  // which is precisely the asymmetry this whole change exists to remove.
  assert.match(probe, /'#\/bank': \{[\s\S]*\{ name: 'caption-lab-picker', open:/)
  assert.match(probe, /'#\/bank': \{[\s\S]*\{ name: 'caption-lab', open:/)
  assert.match(probe, /#bank-passes-panel >> button:has-text\("Caption"\)/)
})

// --- what the adversarial pass caught, pinned so it cannot come back -------------

test('the picker never spells a per-surface field name itself', () => {
  /* THE BUG THIS REPLACES. The picker was written on the dataset shape and read
     `img.filename` in the filter and the aria-label. A bank row has no such key — it
     carries `name` (the basename of its relpath) — so on the Bank the "filter by
     filename" half matched the empty string forever and every label fell back to a row
     id. Nothing failed: the responsive probe matches the label by PREFIX, so an id read
     as a pass. One resolver, and the picker may not reach around it. */
  assert.doesNotMatch(picker, /img\.filename/,
    'the picker reads a dataset-only field again — names go through nameOf')
  assert.match(picker, /nameOf = imageDisplayName/)
  assert.match(picker, /aria-label=\{`Bench captions on image \$\{nameOf\(img\)\}`\}/)
  // The resolver covers BOTH vocabularies, and never returns undefined.
  assert.match(surfaces,
    /imageDisplayName = \(img\) => img\.filename \|\| img\.name \|\| String\(img\.id\)/)
  assert.match(bank, /imageLabel=\{imageDisplayName\(labImage\)\}/)
})

test('the dataset picker asks for tile-sized pictures, not originals', () => {
  /* /img/ serves the ORIGINAL bytes and the picker offers the whole kept pile, so a
     grid of ~200 px cells was pulling megabytes per row. datasetThumbUrl is the app's
     one rewrite for exactly this; every other tile grid already uses it. */
  assert.match(workspace, /thumbUrl=\{\(img\) => datasetThumbUrl\(/)
  assert.match(workspace, /import \{ datasetThumbUrl \}/)
})

test('the topmost layer owns Escape, and an unsaved caption is not thrown away', () => {
  /* On the Bank the bench opens from INSIDE the Caption launch window, which closes on
     Escape too: one press was closing both and taking the run dials with it. And
     '‹ Another image' unmounts the dialog — on a component that exists because a
     refused save once destroyed a caption, it may not do that silently. */
  for (const [name, src] of [['picker', picker], ['dialog', dialog]]) {
    assert.match(src, /stopImmediatePropagation\(\)/,
      `${name}: Escape no longer stops at the topmost layer`)
    assert.match(src, /addEventListener\('keydown', closeOnEscape, true\)/,
      `${name}: the Escape listener left the capture phase`)
  }
  assert.match(dialog, /const dirty = draft !== \(initialCaption \|\| ''\)/)
  assert.match(dialog, /if \(dirty && !leaveArmed\)/)
  // …and the bank tells the user the truth about what a bank caption is for.
  assert.match(bank, /captionPlaceholder="Caption — a plain description, used for search…"/)
})
