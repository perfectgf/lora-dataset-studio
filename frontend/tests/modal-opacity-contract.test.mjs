import test from 'node:test'
import assert from 'node:assert/strict'
import { existsSync, readdirSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

// WHY THIS TEST EXISTS
// --------------------
// The Edit-reference modal shipped see-through: its form sat straight on the dim
// overlay with no panel behind it, so the page bled through the gaps. It was
// fixed — and then LOST when the file was rewritten for the server-job rework,
// because nothing failed when the opaque card vanished (opacity is invisible to
// logic tests). This contract makes that regression loud: every dialog that is a
// FORM/PANEL must give its content an opaque surface, so the next rewrite can't
// quietly drop it.
//
// The trap it guards against: `bg-surface` is only 4% alpha (--surface-alpha) —
// a tint meant to sit ON a solid surface, never to BE one. Opaque panel tokens
// are bg-surface-overlay, bg-surface-solid, bg-app (and a bare, alpha-free
// bg-black card). A dialog that has only a semi-transparent overlay (bg-black/NN)
// and no opaque card is the exact bug.

const HERE = fileURLToPath(new URL('.', import.meta.url))
const COMPONENTS = join(HERE, '..', 'src', 'components')

// Fullscreen IMAGE VIEWERS are exempt: the image itself fills the backdrop and
// is opaque, so they legitimately need no card. Add a file here ONLY when it is
// genuinely an image/media viewer, never to silence a real see-through form.
const IMAGE_VIEWER_ALLOWLIST = new Set([
  'CropModal.jsx',
  'DatasetLightbox.jsx',
  'ResultLightbox.jsx',
  'WatermarkReviewLightbox.jsx',
  'QuickVoteModal.jsx',        // fullscreen bg-black/95 image vote
  'BankReviewLightbox.jsx',    // fullscreen bg-black/95 bank triage viewer
  'PreviewLightbox.jsx',       // thin adapter over GeneratedImageLightbox
])

// Opaque panel tokens (alpha-free surfaces a card can be built on).
const OPAQUE = /\bbg-(surface-overlay|surface-solid|app)\b|\bbg-black(?=["'\s])/

// A dialog's classes do not all live in its own file. When a responsive shape
// gets big enough to reason about, it is extracted to a sibling module of
// CLASS CONSTANTS (`export const FACTS_PANEL_CLASS = '… bg-app …'`) so
// `node --test` can assert the breakpoints without a DOM — and this guard, which
// only ever read .jsx, went blind the first time that happened: the opaque token
// was still there, one import away, and the test called the dialog see-through.
//
// So the scan follows exactly one kind of import: a RELATIVE one whose imported
// bindings are ALL SCREAMING_SNAKE_CASE, which is what a class-constant module
// looks like and what a component or hook does not. One level deep, no
// recursion — enough to see the extracted classes, not enough to let some
// unrelated `bg-app` three files away vouch for a genuinely transparent form.
const CLASS_MODULE_IMPORT = /import\s*\{([^}]*)\}\s*from\s*['"](\.[^'"]*)['"]/g

function withClassModules(file, src) {
  let text = src
  for (const [, names, spec] of src.matchAll(CLASS_MODULE_IMPORT)) {
    const bindings = names.split(',').map((n) => n.trim()).filter(Boolean)
    if (!bindings.length) continue
    if (!bindings.every((n) => /^[A-Z][A-Z0-9_]*$/.test(n))) continue
    for (const ext of ['.js', '.jsx', '']) {
      const p = join(dirname(file), spec + ext)
      if (existsSync(p)) { text += '\n' + readFileSync(p, 'utf8'); break }
    }
  }
  return text
}

function jsxFiles(dir) {
  const out = []
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, entry.name)
    if (entry.isDirectory()) out.push(...jsxFiles(p))
    else if (entry.name.endsWith('.jsx')) out.push(p)
  }
  return out
}

test('every form/panel dialog uses an opaque surface (no see-through modals)', () => {
  const offenders = []
  for (const file of jsxFiles(COMPONENTS)) {
    const src = readFileSync(file, 'utf8')
    if (!src.includes('role="dialog"')) continue
    const name = file.split(/[\\/]/).pop()
    if (IMAGE_VIEWER_ALLOWLIST.has(name)) continue
    if (!OPAQUE.test(withClassModules(file, src))) offenders.push(name)
  }
  assert.deepEqual(offenders, [],
    `these dialogs have no opaque surface token — their content will show the ` +
    `page through it. Give the card bg-surface-overlay (not bg-surface, which is ` +
    `4% alpha), or add it to IMAGE_VIEWER_ALLOWLIST if it is truly an image viewer: ` +
    offenders.join(', '))
})
