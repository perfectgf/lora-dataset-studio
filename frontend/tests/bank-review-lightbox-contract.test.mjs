/* ▶ Review (Bank fast triage) — wiring contract.
 *
 * The navigation itself is unit-tested in src/components/bank/bankReview.test.js.
 * What node --test can't exercise (JSX) is asserted here as text, on the few
 * properties that would silently ruin the mode if a refactor dropped them:
 * the tile click must STAY the bulk-selection gesture, decisions must go out one
 * POST at a time, and a failed POST must not advance.
 */
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'
import { WHATS_NEW } from '../src/whatsNew.js'
import { WHATS_NEW_ARCHIVE } from '../src/whatsNewArchive.js'

// The entry under test may have moved to the archive since it shipped
// (see whatsNew.js, rule "Keep the list tidy") — search the union.
const ALL_WHATS_NEW = [...WHATS_NEW, ...WHATS_NEW_ARCHIVE]
import { getHelpTopic } from '../src/help/helpRegistry.js'
import { REVIEW_SHORTCUT_HINT, ownsTypedKeys } from '../src/components/shared/reviewShortcuts.js'

const here = path.dirname(fileURLToPath(import.meta.url))
const frontend = path.resolve(here, '..')
const read = (rel) => fs.readFileSync(path.join(frontend, rel), 'utf8')
const lightbox = read('src/components/bank/BankReviewLightbox.jsx')
// The review cluster moved to useReviewLightbox (hook wave 5); the
// contract is about the behaviour, so it reads both sources as one text.
const workspace = read('src/components/bank/BankWorkspace.jsx')
  + read('src/components/bank/useReviewLightbox.js')
// The tile is its own component since the Encre redesign; the review WIRING
// (which ids, which start) stays in the workspace.
const tile = read('src/components/bank/BankTile.jsx')

test('the lightbox offers Keep, Reject and Skip with their K/R/S shortcuts', () => {
  assert.match(lightbox, /✓ Keep/)
  assert.match(lightbox, /✕ Reject/)
  assert.match(lightbox, /⏭ Skip/)
  // The keys themselves now come from the SHARED grammar (unit-tested in
  // src/components/shared/reviewShortcuts.test.js) — the same module the
  // dataset lightbox reads, so the two review surfaces cannot drift apart. What
  // is pinned here is that this lightbox still binds each verdict to its own
  // action rather than quietly dropping one.
  assert.match(lightbox, /reviewKeyAction\(e\)/)
  assert.match(lightbox, /action === 'keep'[^\n]*sendDecision\('keep'\)/)
  assert.match(lightbox, /action === 'reject'[^\n]*sendDecision\('reject'\)/)
  assert.match(lightbox, /action === 'skip'[^\n]*doSkip\(\)/)
  assert.match(lightbox, /action === 'back'[^\n]*goBack\(\)/)
  assert.match(lightbox, /action === 'close'[^\n]*onClose\(\)/)
  // Discoverable, not folklore: the shortcuts are printed in the UI, from the
  // same constant the handler is built around.
  assert.match(lightbox, /\{REVIEW_SHORTCUT_HINT\}/)
  assert.equal(REVIEW_SHORTCUT_HINT, 'K keep · R reject · S skip')
})

test('the shortcuts survive the focus trap landing on the 🎲 checkbox', () => {
  // A blanket `tag === 'input'` guard silently killed K/R/S: useFocusTrap
  // focuses the first focusable, which is the Random-order checkbox. The guard
  // moved into the shared module with the rest of the grammar; this pins that
  // the Bank has NOT grown a private copy of it again.
  assert.doesNotMatch(lightbox, /if \(tag === 'input'[^\n]*\) return/)
  assert.doesNotMatch(lightbox, /\['checkbox', 'radio', 'button', 'submit', 'range'\]/)
  assert.equal(ownsTypedKeys({ tagName: 'INPUT', type: 'checkbox' }), false)
  assert.equal(ownsTypedKeys({ tagName: 'TEXTAREA' }), true)
  // …and that the Bank's OWN keys still honour it — [ ] and M are read off the
  // same event after the shared grammar has declined it.
  assert.match(lightbox, /ownsTypedKeys\(e\.target\)\) return/)
})

test('each decision is one immediate POST for one image', () => {
  assert.match(lightbox, /postJson\(`\/api\/bank\/\$\{bankId\}\/images\/status`, \{ ids: \[target\], status \}\)/)
})

test('a failed decision surfaces an error and does NOT advance', () => {
  const send = lightbox.match(/const sendDecision = useCallback\(([\s\S]*?)\n  \}, \[/)[1]
  const [tryPart, catchPart] = send.split('} catch (e) {')
  assert.match(tryPart, /setSession\(\(s\) => decide\(s, status\)\)/)
  assert.doesNotMatch(catchPart, /setSession/)
  assert.match(catchPart, /setError\(/)
})

test('the lightbox renders the FULL file, never the grid thumbnail', () => {
  assert.match(lightbox, /\/api\/bank\/\$\{bankId\}\/file\/\$\{id\}/)
  assert.doesNotMatch(lightbox, /\/thumb\//)
})

test('the end-of-pool state is explicit and closable', () => {
  assert.match(lightbox, /All \{p\.total\.toLocaleString\(\)\} image/)
  assert.match(lightbox, /Back to the grid/)
})

test('random order is a shuffle of the remaining ids, not a random draw', () => {
  const logic = read('src/components/bank/bankReview.js')
  assert.match(logic, /export function setShuffle/)
  assert.match(lightbox, /setShuffle\(s, !s\.shuffle\)/)
  // No "pick an index at random" anywhere in the navigation lane.
  assert.doesNotMatch(logic, /Math\.random\(\)\s*\*\s*(s\.|order)/)
})

test('the workspace opens the review over a SNAPSHOT of the current filter', () => {
  assert.match(workspace, /const openReview = async/)
  assert.match(workspace, /await fetchAllIds\(bankId, filterParams\(filter\)\)/)
  assert.match(workspace, /<BankReviewLightbox bankId=\{bankId\} ids=\{review\.ids\} startId=\{review\.startId\}/)
})

test('the tile click still (de)selects — review is its own ▶ hit target', () => {
  assert.match(workspace, /onReview=\{\(\) => openReview\(img\.id\)\}/)
  assert.match(workspace, /onToggle=\{\(\) => setSelected\(/)
  assert.match(tile, /<button type="button" onClick=\{onReview\}/)
  assert.match(tile, /<button type="button" onClick=\{onToggle\}/)
})

test('header counters follow the run and the grid refreshes on close', () => {
  assert.match(workspace, /const onReviewDecided = \(\) => \{ refreshPayload\(\) \}/)
  assert.match(workspace, /const closeReview = \(\) => \{ setReview\(null\); refreshPayload\(\); refreshImages\(\) \}/)
})

test('the mode is announced and documented', () => {
  const entry = ALL_WHATS_NEW.find((e) => e.id === '2026-07-24-bank-review-one-by-one')
  assert.ok(entry, "What's new entry for the review mode is missing")
  // Archived → no in-app target, by doctrine (whatsNew.js, "Keep the list tidy").
  assert.equal(entry.to, undefined)
  const topic = getHelpTopic('bank-review-one-by-one')
  assert.ok(topic, 'help topic bank-review-one-by-one is missing')
  assert.equal(topic.guide.anchor, 'review-a-bank-one-image-at-a-time')
})
