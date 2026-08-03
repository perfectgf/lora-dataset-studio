/**
 * Contract test for the mobile chip rails — the `relative` that keeps a phone
 * layout full-width.
 *
 * THE BUG THIS PINS. Reported from a phone: every bar on the dataset page drew
 * at roughly 73% of the screen, header included, with dead black space down the
 * right-hand side. Nothing looked broken up close — no clipped text, no visible
 * element sticking out — because the thing sticking out was one pixel wide and
 * deliberately invisible.
 *
 * `overflow-x-auto` clips a descendant only when the scroller is ALSO that
 * descendant's containing block, and a `position: static` box never is. The
 * section chips carry a NavBadge whose count is spelled out for screen readers
 * in a `.sr-only` span — which Tailwind implements as `position: absolute` with
 * no `top`/`left`. With no positioned ancestor it resolves against the document
 * and keeps its STATIC position: out at the far end of a rail 1123 px wide.
 * Measured on the reported page, that put the document at 598 px against a
 * 440 px viewport, so mobile Safari shrank the page to fit — 440/598 ≈ 73.6%,
 * exactly the ratio in the screenshot.
 *
 * One word on the scroller fixes it, and no test that renders nothing can feel
 * the difference: node --test cannot parse JSX and there is no layout engine
 * here, so this file greps the rails for the class. That is the whole defence —
 * a "tidy the classNames" pass that drops `relative` reintroduces a bug whose
 * symptom appears nowhere near the code that caused it.
 */
import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const read = (rel) => readFileSync(new URL(rel, import.meta.url), 'utf8')

// Every horizontally scrolling chip rail: the file, and the rail's aria-label.
const RAILS = [
  ['../src/components/dataset/DatasetWorkspace.jsx', 'Dataset sections'],
  ['../src/pages/SettingsPage.jsx', 'Settings sections'],
  ['../src/pages/GuidePage.jsx', 'Guide chapters'],
]

for (const [file, label] of RAILS) {
  test(`the "${label}" mobile rail is a containing block`, () => {
    const src = read(file)
    // The mobile rail is the one that both scrolls and hides at lg. Match its
    // className wherever the attribute sits relative to aria-label.
    const rails = [...src.matchAll(/className="([^"]*overflow-x-auto[^"]*lg:hidden[^"]*)"/g)]
      .map((m) => m[1])
    assert.ok(rails.length > 0, `${file}: no mobile scrolling rail found — did the markup move?`)
    for (const cls of rails) {
      assert.ok(/(^|\s)relative(\s|$)/.test(cls),
        `${file}: a scrolling mobile rail lost "relative" (${cls.trim()}). An absolutely `
        + 'positioned descendant — an .sr-only label is one — then escapes the scroller and '
        + 'widens the document past the viewport, shrinking the whole phone layout.')
    }
  })
}

test('the destinations rail under the dataset sections is covered too', () => {
  // It carries no badge today, so it cannot be caught by symptom — only by rule.
  const src = read('../src/components/dataset/DatasetWorkspace.jsx')
  assert.match(src, /className="relative -mx-4 -mt-1 overflow-x-auto px-4 pb-3 lg:hidden"/)
})

test('the reason is written down where the class is', () => {
  // A bare `relative` reads as noise and gets tidied away. The explanation is
  // the only thing that makes it survive the next pass over this markup.
  const src = read('../src/components/dataset/DatasetWorkspace.jsx')
  assert.match(src, /containing block/,
    'the dataset rail must keep the note explaining why `relative` is load-bearing')
})
