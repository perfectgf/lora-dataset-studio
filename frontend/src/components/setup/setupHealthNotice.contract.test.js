/* The onboarding redirect acts on the server's answer, never on a request
   that failed. Text contracts on the two files that decide it: the source
   tests cannot mount the router, and the guard below is what turned a phone's
   reconnecting link into a detour through Setup (2026-09-04). */
import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'

const NOTICE = fs.readFileSync(new URL('./SetupHealthNotice.jsx', import.meta.url), 'utf8')
const CAPS = fs.readFileSync(new URL('../../context/CapabilitiesContext.jsx', import.meta.url), 'utf8')

test('a failed setup-state request is retried once, then ignored — never read as "never verified"', () => {
  assert.doesNotMatch(NOTICE, /s = \{ verified: false, checks: \{\}, regressions: \[\] \}/,
    'the fallback that turned a dropped request into a first run must stay gone')
  assert.match(NOTICE, /for \(let attempt = 0; attempt < 2 && alive; attempt \+= 1\) \{\n\s*try \{\n\s*s = await apiFetch\('\/api\/setup-state', \{ background: true \}\)\n\s*break/)
  assert.match(NOTICE, /if \(!alive \|\| !s\) return\n\s*setState\(s\)/)
})

test('the notice waits for capabilities that ANSWERED, and hands the rule that fact', () => {
  assert.match(NOTICE, /const \{ caps, loading, known, refresh \} = useCapabilities\(\)/)
  assert.match(NOTICE, /if \(loading \|\| !known \|\| startedRef\.current\) return/)
  assert.match(NOTICE, /capsKnown: known, state: s, pendingDockerChoice,/)
  assert.match(NOTICE, /\}, \[loading, known, caps, navigate, pathname, refresh\]\)/)
})

test('the capabilities context says whether its answer is real, and retries the first load once', () => {
  assert.match(CAPS, /const \[known, setKnown\] = useState\(false\)/)
  assert.match(CAPS, /setKnown\(true\)\n\s*setCaps\(data\)\n\s*return data/)
  assert.match(CAPS, /value=\{\{ caps, loading, known, refresh \}\}/)
  assert.match(CAPS, /if \(\(await refresh\(\)\) !== null \|\| !alive\) return\n[\s\S]*?refresh\(false, \{ background: true \}\)/)
})
