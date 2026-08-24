/**
 * The image Bank's component tree, as one string, for the WIRING tests.
 *
 * Why this exists. Eighteen test files pinned their assertions to
 * `BankWorkspace.jsx` — the file — back when the whole screen was one 3 471-line
 * component. That is the very trap bankSurfaces.js was built to avoid and says
 * so in its own header: "a test that asserted a label in a given FILE would fail
 * on every move and teach us to weaken it". The surface inventory got the
 * lesson; these did not, so the Encre redesign turned 31 of them red for the one
 * reason that is not a regression — the code moved.
 *
 * So they read the TREE instead. Every assertion keeps exactly the strength it
 * had (the wiring must still exist, somewhere a user can reach it); only the
 * question "in which file" is dropped, which was never the thing being
 * protected. Splitting a component is now free; deleting the wiring is still
 * loud.
 *
 * ⚠️ Scoped to the IMAGE bank on purpose — `components/videobank` is NOT
 * included. The video lane shares surfaces with this one, so folding it in
 * would let a pattern satisfied over there mark an image-bank test green while
 * the image bank had lost it. That is the exact failure the surface inventory
 * hit once already (its "Open →" collision), and it is why this helper is
 * narrower than the contract test's own scan.
 *
 * Files are DISCOVERED, not listed, so a file the next refactor creates is
 * covered without anyone remembering to add it here.
 */
import { readFileSync, readdirSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const FRONTEND = resolve(HERE, '../../..')

/**
 * BankWorkspace.jsx alone — for assertions that SLICE a handler body out of the
 * source (`src.slice(src.indexOf('const passBody'), src.indexOf('const runPass'))`).
 *
 * ⚠️ Those must NOT read the tree. A positional slice needs its two markers to be
 * unique and ordered, and over a concatenation they are neither:
 * BankThresholdsPanel.jsx also declares a `const runPass`, and it sorts before
 * BankWorkspace.jsx, so the end marker was found ahead of the start one and the
 * slice came back empty — a test asserting against '' fails loudly here, but the
 * same shape could just as easily have gone quietly green.
 *
 * Pinning them to this file is right rather than merely expedient: the redesign
 * moves JSX out of the workspace, while the state and the request builders stay.
 * These assertions are about the handler layer, and the handler layer is exactly
 * what does not move.
 */
export function bankWorkspaceSource() {
  return readFileSync(resolve(HERE, 'BankWorkspace.jsx'), 'utf8').replace(/\r\n/g, '\n')
}

/** Every .jsx of the image Bank, plus the page that mounts it, concatenated and
 *  newline-normalised (a CRLF checkout must not change what a regex sees). */
export function bankTreeSource() {
  const parts = readdirSync(HERE, { recursive: true })
    .map(String)
    // The hook layer the decomposition creates (use*.js) is part of the
    // tree: state and handlers moved there, and the wiring contract is
    // 'somewhere a user can reach it', not 'in which file'. Pure-logic
    // helpers (bankSync.js, bankReview.js...) stay excluded on purpose.
    .filter((f) => f.endsWith('.jsx') || /use[A-Z]\w*\.js$/.test(f))
    .sort()
    .map((f) => readFileSync(resolve(HERE, f), 'utf8'))
  parts.push(readFileSync(resolve(FRONTEND, 'src/pages/BankPage.jsx'), 'utf8'))
  return parts.join('\n').replace(/\r\n/g, '\n')
}
