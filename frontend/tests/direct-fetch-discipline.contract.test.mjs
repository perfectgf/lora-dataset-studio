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

// Every raw fetch('/api…') call site, with enough trailing context to see the
// options object it was called with.
const directCalls = (text) => {
  const out = []
  const re = /\bfetch\((['`])\/api/g
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

test('the set of files allowed to raw-fetch is closed', () => {
  // Each entry is a deliberate best-effort GET (or the client itself). Adding
  // a file here needs the same justification these carry at the call site.
  const ALLOWED = new Set([
    'api/fetchClient.js',                                  // the client's own transport
    'App.jsx',                                             // boot health probe, before toasts exist
    'utils/extensionLoader.js',                            // isolated by design: extensions must not inherit app plumbing
    'utils/connectionStatus.js',                           // the offline indicator's own probe
    'hooks/useDataset.js',                                 // legacy raw GETs with local error handling
    'hooks/useLoraTestStudio.js',
    'hooks/useStudioRun.js',
    'pages/CloudRunsPage.jsx',
    'components/dataset/CloudLaunchDialog.jsx',
    'components/dataset/PublishHfModal.jsx',
    'components/dataset/TrainingPanel.jsx',
    'components/dataset/useTrainingPresets.js',          // inherited the panel's best-effort preset-list GET (hook wave 1)
    'components/dataset/TrainingProgress.jsx',
    'components/dataset/TrainingReadiness.jsx',
    'components/dataset/VariationCatalog.jsx',
    'components/dataset/studio/LoraPicker.jsx',
    'components/dataset/studio/StudioGenerationSettings.jsx',
    'components/dataset/studio/StudioRunSetup.jsx',
    'components/dataset/studio/StudioShell.jsx',
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
