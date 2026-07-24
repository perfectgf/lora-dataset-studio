import test from 'node:test'
import assert from 'node:assert/strict'
import { readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
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
])

// Opaque panel tokens (alpha-free surfaces a card can be built on).
const OPAQUE = /\bbg-(surface-overlay|surface-solid|app)\b|\bbg-black(?=["'\s])/

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
    if (!OPAQUE.test(src)) offenders.push(name)
  }
  assert.deepEqual(offenders, [],
    `these dialogs have no opaque surface token — their content will show the ` +
    `page through it. Give the card bg-surface-overlay (not bg-surface, which is ` +
    `4% alpha), or add it to IMAGE_VIEWER_ALLOWLIST if it is truly an image viewer: ` +
    offenders.join(', '))
})
