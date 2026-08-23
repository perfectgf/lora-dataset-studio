import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const source = readFileSync(
  new URL('./LocalToolsSection.jsx', import.meta.url), 'utf8')
const primitives = readFileSync(
  new URL('./primitives.jsx', import.meta.url), 'utf8')
const settingsPage = readFileSync(
  new URL('../../pages/SettingsPage.jsx', import.meta.url), 'utf8')
// The cloud dialog's token notices moved to CloudLaunchDialog.jsx (panel
// decomposition slice 1); the contract spans both sources.
const trainingPanel = readFileSync(
  new URL('../dataset/TrainingPanel.jsx', import.meta.url), 'utf8')
  + readFileSync(
    new URL('../dataset/CloudLaunchDialog.jsx', import.meta.url), 'utf8')

function handleSaveSource() {
  const start = settingsPage.indexOf('  const handleSave = async () => {')
  const end = settingsPage.indexOf('\n  // Dirty =', start)
  assert.ok(
    start >= 0 && end > start,
    'SettingsPage handleSave is present and bounded by the dirty-state section',
  )
  return settingsPage.slice(start, end)
}


test('classic Hugging Face token links directly to read-token creation', () => {
  const classicTokenGuide = source.match(
    /<a\b[^>]*>\s*Create a read token on Hugging Face ↗\s*<\/a>/)
  assert.ok(classicTokenGuide, 'the classic Hugging Face token guide link is present')
  assert.match(classicTokenGuide[0], /href="https:\/\/huggingface\.co\/settings\/tokens\/new\?tokenType=read"/)
  assert.match(classicTokenGuide[0], /target="_blank"/)
  assert.match(classicTokenGuide[0], /rel="noreferrer"/)
})

test('HF permission requirements are visually emphasized', () => {
  const classicSecret = source.match(/const HF_SECRET = \{[\s\S]*?\n\}/)
  assert.ok(classicSecret, 'the classic Hugging Face secret descriptor is present')
  assert.match(
    classicSecret[0],
    /create a token with the\{' '\}\s*<strong className="font-semibold text-content">read<\/strong>\s*\{' role\./,
  )

  const cloudSecret = source.match(/const HF_CLOUD_SECRET = \{[\s\S]*?\n\}/)
  assert.ok(cloudSecret, 'the dedicated cloud secret descriptor is present')
  const strongText = /<strong className="font-semibold text-content">([^<]+)<\/strong>/g
  assert.deepEqual(
    [...classicSecret[0].matchAll(strongText)].map((match) => match[1]),
    ['read'],
  )
  assert.deepEqual(
    [...cloudSecret[0].matchAll(strongText)].map((match) => match[1]),
    [
      'repo.content.read exactly on krea/Krea-2-Raw',
      'repo.content.read + repo.write on one dedicated HF user/org namespace that contains only LDS deliveries',
    ],
  )
})

test('dense cloud uses its own clearly scoped HF secret field', () => {
  assert.match(source, /key:\s*'HF_CLOUD_TOKEN'/)
  assert.match(source, /key:\s*'HF_CLOUD_TOKEN',[^\n]*testTarget:\s*'hf_cloud'/)
  assert.match(source, /Dedicated Hugging Face cloud token/)
  assert.match(source, /separate fine-grained token/)
  assert.match(source, /zero global permissions/)
  assert.match(source, /repo\.content\.read exactly on krea\/Krea-2-Raw/)
  assert.match(source, /repo\.content\.read \+ repo\.write on one dedicated HF user\/org namespace/)
  assert.match(source, /contains only LDS deliveries/)
  assert.match(source, /per-run repository does not exist yet/)
  assert.doesNotMatch(source, /only its private delivery repositories/)
  assert.match(source, /A global write token is also accepted/)
  const cloudTokenGuide = source.match(
    /<a\b[^>]*>\s*Create a fine-grained token on Hugging Face ↗\s*<\/a>/)
  assert.ok(cloudTokenGuide, 'the dedicated cloud-token guide link is present')
  assert.match(cloudTokenGuide[0], /href="https:\/\/huggingface\.co\/settings\/tokens\/new\?tokenType=fineGrained"/)
  assert.match(cloudTokenGuide[0], /target="_blank"/)
  assert.match(cloudTokenGuide[0], /rel="noreferrer"/)
  const globalWriteGuide = source.match(
    /<a\b[^>]*>\s*Create a global write token on Hugging Face ↗\s*<\/a>/)
  assert.ok(globalWriteGuide, 'the global write-token guide link is present')
  assert.match(globalWriteGuide[0], /tokenType=write/)
  assert.match(globalWriteGuide[0], /target="_blank"/)
  assert.match(globalWriteGuide[0], /rel="noreferrer"/)
  assert.match(source, /<SecretField field=\{HF_CLOUD_SECRET\}/)
  assert.match(source, /<SecretField field=\{HF_SECRET\}/)
})

test('saving a pending HF cloud token consumes the PUT validation inline', () => {
  const handleSave = handleSaveSource()
  const catchStart = handleSave.indexOf('    } catch (e) {')
  const successBody = handleSave.slice(0, catchStart)

  assert.match(handleSave, /Object\.prototype\.hasOwnProperty\.call\(secrets, 'HF_CLOUD_TOKEN'\)/)
  assert.equal(
    (handleSave.match(/putJson\('\/api\/settings'/g) || []).length,
    1,
    'Save performs one settings PUT and no validation request of its own',
  )
  assert.match(successBody, /data\?\.secret_checks\?\.HF_CLOUD_TOKEN/)
  assert.match(successBody, /recordTestResult\('hf_cloud', cloudTokenCheck\)/)
  assert.match(successBody, /recordTestResult\('hf_cloud', cloudTokenCheck\)[\s\S]*toast\.success\('Settings saved\. Dedicated Hugging Face cloud token validated\.'\)/)
  assert.match(successBody, /cloudTokenCheck\?\.code === 'broad_access'/)
  assert.match(successBody, /cloudTokenCheck\?\.severity === 'warning'/)
  assert.match(successBody, /toast\.warning\(/)
  assert.match(successBody, /cloudTokenCheck\.warning/)
  assert.match(successBody, /else \{\s*toast\.success\('Settings saved\.'\)/)
  assert.doesNotMatch(handleSave, /postJson|\/api\/settings\/test\/hf_cloud/)
})

test('a structured HF cloud validation failure stays inline and preserves the input', () => {
  const handleSave = handleSaveSource()
  const catchStart = handleSave.indexOf('    } catch (e) {')
  const finallyStart = handleSave.indexOf('    } finally {', catchStart)
  const catchBody = handleSave.slice(catchStart, finallyStart)

  assert.match(catchBody, /e\?\.body\?\.secret_checks\?\.HF_CLOUD_TOKEN/)
  assert.match(catchBody, /recordTestResult\('hf_cloud', cloudTokenCheck\)/)
  assert.match(catchBody, /toast\.error\(\`Dedicated Hugging Face cloud token was not saved: \$\{detail\}\`\)/)
  assert.doesNotMatch(catchBody, /setSecretInputs/)
  assert.doesNotMatch(catchBody, /toast\.success/)
  assert.doesNotMatch(
    handleSave,
    /toast\.(?:success|error)\([^)]*(?:secretInputs|secrets)/s,
  )
})

test('HF validation results expose success, warning, and error without relying on color', () => {
  assert.match(primitives, /result\.severity === 'warning' \|\| result\.code === 'broad_access'/)
  assert.match(primitives, /success: \{ glyph: '\\u2713'.*text-emerald-400/)
  assert.match(primitives, /warning: \{ glyph: '\\u26A0'.*text-amber-400/)
  assert.match(primitives, /error: \{ glyph: '\\u2717'.*text-rose-400/)
  assert.match(primitives, /role=\{level === 'error' \? 'alert' : 'status'\}/)
  assert.match(primitives, /<span className="sr-only">\{presentation\.label\}: <\/span>\{detail\}/)
})

test('full-model notices recommend scoped access while accepting global write', () => {
  assert.doesNotMatch(trainingPanel, /requires a fine-grained <code>HF_CLOUD_TOKEN<\/code>/)
  assert.doesNotMatch(trainingPanel, /uploaded using a fine-grained <code>HF_CLOUD_TOKEN<\/code>/)
  assert.equal((trainingPanel.match(/fine-grained token is recommended/g) || []).length, 2)
  assert.equal(
    (trainingPanel.match(/global\s+write token is also\s+accepted with a warning/g) || []).length,
    2,
  )
})
test('focus=HF_CLOUD_TOKEN lands on the secret input id', () => {
  assert.match(primitives, /id=\{f\.key\}/)
  assert.match(primitives, /htmlFor=\{f\.key\}/)
  assert.match(primitives, /type="password"/)
  assert.match(primitives, /\{f\.testTarget && <TestResult result=\{testResults\[f\.testTarget\]\} \/>\}/)
  assert.match(primitives, /onResult\(await postJson\(\`\/api\/settings\/test\/\$\{target\}\`, \{\}\)\)/)
})
