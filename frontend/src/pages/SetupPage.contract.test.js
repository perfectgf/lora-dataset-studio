/**
 * The Setup wizard's Ollama step, held to what it promises — read as source text.
 *
 * Why this file exists: the whole "Setup no longer stops at Ollama" change hangs on a
 * handful of lines inside SetupPage.jsx, and `node --test` renders no JSX. Every other
 * test for this wave covers the PURE helpers in useSetupSteps.js — the gate, the lists.
 * Reverting the one line in `nextWithSave` that opens the skip panel would restore the
 * original wall with the backend suite and the frontend suite both entirely green.
 *
 * Source-as-text is the only instrument `node --test` has here (see
 * settingDefaults.js and LocalToolsSection.contract.test.js for the same technique).
 * It proves a line is WRITTEN, never that the screen works — the runtime proof is the
 * headless walkthrough. So these assertions are deliberately about presence and
 * ORDER, the two properties that were actually got wrong and fixed.
 */
import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const source = readFileSync(new URL('./SetupPage.jsx', import.meta.url), 'utf8')

test('the Ollama step offers a conscious way out, and persists it', () => {
  // The panel and its button, asserted as FACTS rather than as literal text: the
  // provider name became an expression when LM Studio arrived, and a contract that
  // breaks on a behaviour-identical rewording is a contract nobody trusts. What
  // must hold is that the panel exists, that it names the configured provider, and
  // that its button commits through skipOllama.
  assert.match(source, /Continue without \{llmName\}\?/,
    'the confirmation panel that lists what turns off is gone')
  assert.match(source, /const llmName = step\.isLmStudio/,
    'the panel names Ollama unconditionally again — wrong product under LM Studio')
  assert.match(source, /onClick=\{skipOllama\}/,
    'the button that commits the choice no longer calls skipOllama')
  assert.match(source, /Continue without \$\{llmName\}/,
    'the commit button lost its label')
  assert.match(source, /setup_skipped:\s*true/,
    'skipOllama no longer persists the choice')
  assert.match(source, /config:\s*\{\s*ollama:\s*\{\s*setup_skipped:\s*true\s*\}\s*\}/,
    'the PUT no longer writes the ollama section')
})

test('a blocked Next opens the panel instead of only refusing', () => {
  // This is the line the whole fix hangs on. Without it, Next toasts and stays.
  assert.match(source, /if \(reason && s && !s\.dockerManaged\) \{ setOllamaSkipConfirm\(true\); return \}/,
    'nextWithSave no longer opens the skip panel on a native install')
  // ...and the refusal is still ANNOUNCED. The toast carries aria-live; a panel that
  // appears far above the focused button announces nothing on its own.
  const branch = source.slice(source.indexOf('const nextWithSave'))
  const toastAt = branch.indexOf('toast.warning(reason)')
  const openAt = branch.indexOf('setOllamaSkipConfirm(true)')
  assert.ok(toastAt > -1 && openAt > -1, 'the ollama branch of nextWithSave is gone')
  assert.ok(toastAt < openAt,
    'the toast must fire before/alongside the panel, so the refusal is still spoken')
})

test('the panel is an INSERT, never a replacement for the step body', () => {
  // The first cut returned the panel in place of the body, which took the very button
  // the gate names ("click ▶ Start Ollama below", "Pull the vision model below") off
  // the screen. The exit and the remedy are not alternatives.
  assert.doesNotMatch(source, /if \(ollamaSkipConfirm\) return ollamaSkipPanel/,
    'the panel is replacing the step body again — the remedy it names disappears with it')
  assert.match(source, /\{ollamaSkipConfirm && ollamaSkipPanel\}/,
    'the panel is no longer composed above the step body')
  assert.match(source, /\{ollamaBody\}/, 'the step body is no longer rendered alongside the panel')
  assert.match(source, /role="status" aria-live="polite"[\s\S]{0,200}Continue without \{llmName\}\?/,
    'the panel no longer announces itself to a screen reader')
})

test('both skip panels are dismissed when the user navigates between screens', () => {
  // The ComfyUI twin had this from the start; the Ollama one was added without it and
  // came back stale — on top of every state branch, hiding a now-running Ollama.
  assert.match(source, /useEffect\(\(\) => \{ setSkipConfirm\(false\); setOllamaSkipConfirm\(false\) \}, \[screen\]\)/,
    'a half-open skip panel survives navigation and re-opens stale')
})

test('a conscious skip counts as a settled step', () => {
  assert.match(source, /\['ready', 'skipped'\]\.includes\(stepById\[id\]\.status\)/,
    "isReady ignores 'skipped' again — the wizard keeps sending the user back to the step they closed")
})

test('the Docker body never describes Ollama using LM Studio’s readiness', () => {
  // Under LM Studio the native path returns early, but the DOCKER path fell
  // through to three branches that all key on `reachable` — which now means "LM
  // Studio answers". They rendered "✓ Ollama is running at http://ollama:11434"
  // and a ▶ Start Ollama button on installs where Ollama had never run.
  // Order is the assertion: the provider branch has to come FIRST.
  // Anchored on the Ollama step's own body: `if (step.reachable) {` also appears
  // in the ComfyUI step far above, and an unanchored indexOf measured THAT one —
  // the assertion passed on a fresh tree and would have passed on a broken one.
  const body = source.slice(source.indexOf('const lmStudioNote'))
  const guard = body.indexOf('if (step.isLmStudio) {')
  const reachable = body.indexOf('if (step.reachable) {')
  assert.ok(guard > -1, 'the Docker body no longer branches on the provider')
  assert.ok(reachable > -1 && guard < reachable,
    'the Ollama status branches run first again — they describe the wrong server')
  // ...and the cards stay, because the BAT launcher waits on that choice.
  const branch = body.slice(guard, reachable)
  assert.match(branch, /\{deploymentCards\}/,
    'the deployment cards left the page — a Docker user can no longer answer the launcher')
  assert.match(branch, /\{lmStudioNote\}/, 'the body no longer says which server is in use')
})

test('the welcome scan row names the provider it scanned', () => {
  assert.match(source, /label: `Captioning — \$\{llmLabel\} \+ vision model`/,
    'the scan row is hard-coded to Ollama again')
  assert.match(source, /const llmLabel = oll\.isLmStudio/,
    'the label is no longer derived from the active provider')
})

test('an installed Ollama outranks the skip on the welcome scan row', () => {
  // "installed — not running" is actionable and true; "you chose to skip" would hide
  // the ▶ Start button from someone who skipped first and installed Ollama later.
  assert.match(source, /\(oll\.skipped && !oll\.installed\)/,
    'the scan row reports a skip over an installed-but-stopped Ollama')
})
