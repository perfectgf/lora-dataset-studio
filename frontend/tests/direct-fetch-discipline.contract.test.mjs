// Direct fetch('/api…') discipline — the shared client is the only mutation path.
//
// `api/fetchClient.js` carries what every request owes the user: the one-shot
// CSRF-expiry recovery, the offline indicator, and the shared error wording.
// A RAW fetch has none of that, so it is allowed exactly one job: a
// best-effort GET whose failure is expected weather — a health probe during a
// restart (the server is DOWN by design and must not flap the offline
// banner), a cosmetic info line, a capability peek. Two DELETEs used to ride
// raw fetch and paid for it: a stale CSRF token failed them outright where
// every other mutation quietly retries once.
//
// Rule 1: a raw fetch('/api…') must never carry a mutating method.
// Rule 2: the set of files allowed to raw-fetch at all is closed — a new
//         caller either uses the client or names itself here, with a reason.
import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync, readdirSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

const walk = (dirUrl) => {
  const out = []
  for (const entry of readdirSync(dirUrl, { withFileTypes: true })) {
    const child = new URL(`${entry.name}${entry.isDirectory() ? '/' : ''}`, dirUrl)
    if (entry.isDirectory()) out.push(...walk(child))
    else if (/\.jsx?$/.test(entry.name) && !/\.test\.jsx?$/.test(entry.name)) {
      out.push([fileURLToPath(child), readFileSync(fileURLToPath(child), 'utf8')])
    }
  }
  return out
}
const SRC_FILES = walk(new URL('../src/', import.meta.url))

const rel = (path) => path.replace(/\\/g, '/').split('/src/')[1]

// Every raw fetch(...) call site - literal '/api' URL OR a variable/derived
// one - with enough trailing context to see the options object it was
// called with. The variable form used to be a blind spot: postTrain sent
// every training POST through fetch(path) and no rule ever saw it.
// The lookbehind keeps fetchWithCsrfRetry and method calls out of the net.
const directCalls = (text) => {
  const out = []
  const re = /(?<![.\w])fetch\(/g
  for (let m = re.exec(text); m; m = re.exec(text)) {
    out.push({
      line: text.slice(0, m.index).split('\n').length,
      context: text.slice(m.index, m.index + 400),
    })
  }
  return out
}

test('a raw fetch never mutates — every POST/PUT/DELETE/PATCH rides the shared client', () => {
  const offenders = []
  for (const [path, text] of SRC_FILES) {
    if (rel(path) === 'api/fetchClient.js') continue
    for (const call of directCalls(text)) {
      if (/method:\s*['"`](POST|PUT|DELETE|PATCH)/.test(call.context)) {
        offenders.push(`${rel(path)}:${call.line}`)
      }
    }
  }
  assert.deepEqual(offenders, [],
    'mutating requests owe the client its CSRF retry and error wording — use postJson/putJson/patchJson/del/postForm')
})

// Rule 3: a MULTIPART body carries the CSRF token, or it never reaches the view.
//
// This one was not theory. The video dataset's References section shipped with
// `apiFetch(url, { method: 'POST', body: form })` — the only FormData in the app
// that did not go through `postForm` — and CSRFProtect refused it with a 400 the
// view never saw. The backend suite could not catch it (conftest builds the app
// with WTF_CSRF_ENABLED=False, so every machine caller is unverified there), and
// rules 1 and 2 above could not either: it was a client call, not a raw fetch,
// and the client only adds the token for the helpers that build the body.
//
// Anchored on the PAYLOAD, not on the verb, and at the scale of the FILE, not
// of a window of characters. The first version keyed on a literal
// `method: 'POST'` with a `body:` within 300 characters, and four ways of
// writing a token-less multipart POST walked through it — an options object
// built in two statements, a shorthand property, a helper taking the method as
// an argument, and, purely by accident, a long comment inside the options
// object pushing `body:` out of the window. This codebase comments inside
// objects all the time; the corrected call site carries eight lines of it.
//
// What a FormData needs is not a shape, it is a TOKEN. So: a file that
// constructs one must, somewhere in that same file, either hand it to the
// helper that attaches the token (`postForm(`) or set `X-CSRFToken` by hand
// (useDataset.js and DescribeImageModal.jsx do, correctly). No window, no
// verb, nothing a formatting choice can slip past.
// COMMENTS ARE STRIPPED FIRST, and that is measured, not tidy: the corrected
// call site carries an eight-line comment that names `X-CSRFToken` while
// explaining the bug, and a mutation back to the token-less apiFetch stayed
// green because the rule found the header's name in that comment. Code is what
// sends the request; only code counts. And the header has to appear as a KEY
// (`'X-CSRFToken':`), the one shape that actually puts it on the wire.
const codeOnly = (text) => text
  .replace(/\/\*[\s\S]*?\*\//g, '')
  .replace(/^\s*\/\/.*$/gm, '')
const buildsFormData = (text) => /\bnew\s+FormData\s*\(/.test(codeOnly(text))
const carriesToken = (text) => {
  const code = codeOnly(text)
  return /\bpostForm\s*\(/.test(code) || /['"]X-CSRFToken['"]\s*:/.test(code)
}

test('a file that builds a FormData sends it with the CSRF token, whatever the call looks like', () => {
  const offenders = []
  for (const [path, text] of SRC_FILES) {
    if (rel(path) === 'api/fetchClient.js') continue      // postForm lives there
    if (!buildsFormData(text)) continue
    if (carriesToken(text)) continue
    offenders.push(rel(path))
  }
  assert.deepEqual(offenders, [],
    'a FormData without the CSRF token is refused with a 400 the view never sees — send it through postForm')
})

test('a comment that mentions the token does not count as sending it', () => {
  const text = "// the retry is keyed on an X-CSRFToken header — postForm( would set it\n"
    + "const form = new FormData(); apiFetch(url, { method: 'POST', body: form })"
  assert.ok(buildsFormData(text))
  assert.equal(carriesToken(text), false, 'prose about the token is not the token')
  assert.equal(carriesToken("headers: { 'X-CSRFToken': getCsrfToken() }"), true)
  assert.equal(carriesToken('const r = await postForm(url, form)'), true)
})

test('the FormData rule catches every shape that slipped past its first version', () => {
  // Guard of the guard: the four escapes measured on 2026-09-02, as fixtures.
  // Each is a token-less multipart POST and each MUST be an offender.
  const escapes = [
    "const form = new FormData(); const opts = { method: 'POST' }; opts.body = form; apiFetch(url, opts)",
    "const form = new FormData(); const method = 'POST'; apiFetch(url, { method, body: form })",
    "const form = new FormData(); const send = (u, method, body) => apiFetch(u, { method, body }); send(url, 'POST', form)",
    `const form = new FormData(); apiFetch(url, { method: 'POST', /* ${'x'.repeat(480)} */ body: form })`,
  ]
  for (const text of escapes) {
    assert.ok(buildsFormData(text) && !carriesToken(text),
      `an escape is not caught: ${text.slice(0, 60)}…`)
  }
  // …and the two legal shapes stay legal.
  assert.equal(carriesToken('const fd = new FormData(); postForm(url, fd)'), true)
  assert.equal(carriesToken("const fd = new FormData(); fetch(u, { headers: { 'X-CSRFToken': t }, body: fd })"), true)
})

test('the set of files allowed to raw-fetch is closed', () => {
  // Each entry is a deliberate best-effort GET (or the client itself). Adding
  // a file here needs the same justification these carry at the call site.
  const ALLOWED = new Set([
    'api/fetchClient.js',                                  // the client's own transport
    'App.jsx',                                             // boot health probe, before toasts exist
    'utils/extensionLoader.js',                            // isolated by design: extensions must not inherit app plumbing
    'utils/connectionStatus.js',                           // the offline indicator's own probe
    'hooks/useDataset.js',                                 // legacy raw GETs with local error handling
    'hooks/useImageDownload.js',                           // blob download - needs the raw Response, own error copy
    'utils/galleryDownload.js',                            // same blob download, looped: raw Response for Content-Disposition + blob; a miss SKIPS by design
    'hooks/useLoraTestStudio.js',                          // status poll, silent retry on transient errors
    'hooks/useStudioRun.js',                               // run-status poll, silent retry on transient errors
    'pages/CloudRunsPage.jsx',                             // history/lineage GETs render their own error states
    'components/dataset/CloudLaunchDialog.jsx',            // readiness/offers polls inside the dialog, own copy
    'components/dataset/PublishHfModal.jsx',               // whoami/status best-effort, null on failure
    'components/dataset/ConceptFaceMaskField.jsx',        // best-effort mask preview GET, null on failure
    'components/dataset/TrainingPanel.jsx',                // status poll + preflight GET with local handling
    'components/dataset/useTrainingPresets.js',          // inherited the panel's best-effort preset-list GET (hook wave 1)
    'components/dataset/TrainingProgress.jsx',             // progress poll on a timer - silence IS the contract
    'components/dataset/TrainingReadiness.jsx',            // preflight GET where a 409 is an answer, not an error
    'components/dataset/VariationCatalog.jsx',             // catalog GET with its own error state
    'components/dataset/studio/LoraPicker.jsx',            // load-once GET, own empty/error states
    'components/dataset/studio/StudioGenerationSettings.jsx', // config peek, null on failure
    'components/dataset/studio/StudioRunSetup.jsx',        // recent-prompts GET, best-effort
    'components/dataset/studio/StudioShell.jsx',           // base-models GET, own error state
    'components/settings/EnginesSection.jsx',              // capability peek, null on failure
    'components/settings/MaintenanceSection.jsx',          // health poll DURING restart — offline banner must not flap
    'components/settings/ServerSection.jsx',               // same restart poll
    'components/settings/TrainingSection.jsx',             // best-effort info line, says so in place
  ])
  const strays = []
  for (const [path, text] of SRC_FILES) {
    if (directCalls(text).length && !ALLOWED.has(rel(path))) strays.push(rel(path))
  }
  assert.deepEqual(strays, [],
    'new raw fetch("/api…") caller — use api/fetchClient.js, or justify an allowlist entry here')
})
