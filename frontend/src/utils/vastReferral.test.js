import assert from 'node:assert/strict'
import { readFileSync, readdirSync, statSync } from 'node:fs'
import { dirname, join, relative, resolve } from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

import { VAST_CONSOLE_URL, VAST_REFERRAL_ID, vastSignupUrl } from './vastReferral.js'

/* The referral link is a material interest disclosed next to ONE link (the
 * "create an account" step), with the untagged link beside it. Everything
 * below pins the shape that keeps that honest:
 *   · the id lives in vastReferral.js and nowhere else in the sources;
 *   · exactly one component builds the tagged link;
 *   · the escape hatches (Billing, Keys, instances console) stay untagged;
 *   · README and the settings guide carry the SAME id, once each, with the
 *     disclosure and the untagged link — or no id at all when none is set.
 * The rendered guide itself is covered by tests/vast-key-guide-render.test.mjs. */

const HERE = dirname(fileURLToPath(import.meta.url))
const SRC = resolve(HERE, '..')             // frontend/src
const REPO = resolve(SRC, '../..')          // repo root
const read = (abs) => readFileSync(abs, 'utf8')

const TAGGED = /https:\/\/cloud\.vast\.ai\/\?ref_id=([^)\s"'<>&]+)/g
/* The bare console root: not followed by a query string or a sub-path. */
const UNTAGGED = /https:\/\/cloud\.vast\.ai\/(?![?a-z])/

function walk(dir, keep, out = []) {
  for (const name of readdirSync(dir)) {
    if (name === 'node_modules' || name === 'dist') continue
    const abs = join(dir, name)
    if (statSync(abs).isDirectory()) walk(abs, keep, out)
    else if (keep(abs)) out.push(abs)
  }
  return out
}

const SOURCES = walk(SRC, (f) => /\.(jsx?|mjs)$/.test(f) && !/\.test\.(jsx?|mjs)$/.test(f))
const DOCS = walk(resolve(REPO, 'docs'), (f) => f.endsWith('.md'))
const README = resolve(REPO, 'README.md')
const GUIDE = resolve(REPO, 'docs/guide/settings-reference.md')

const rel = (abs) => relative(SRC, abs).replace(/\\/g, '/')
const taggedIds = (text) => [...text.matchAll(TAGGED)].map((m) => m[1])

test('no id → the untagged console; an id → exactly one ref_id on the console root', () => {
  assert.equal(vastSignupUrl(''), VAST_CONSOLE_URL)
  assert.equal(vastSignupUrl(null), VAST_CONSOLE_URL)
  assert.equal(vastSignupUrl('   '), VAST_CONSOLE_URL)
  assert.equal(vastSignupUrl(undefined), vastSignupUrl(VAST_REFERRAL_ID))
  assert.equal(vastSignupUrl(' 12345 '), 'https://cloud.vast.ai/?ref_id=12345')
  assert.equal(vastSignupUrl(12345), 'https://cloud.vast.ai/?ref_id=12345')
  assert.equal(vastSignupUrl('a b&c'), 'https://cloud.vast.ai/?ref_id=a%20b%26c')
})

test('the sources never spell a tagged vast.ai link out — vastSignupUrl is the only builder', () => {
  const offenders = SOURCES.filter((f) => TAGGED.test(read(f)) || /ref_id=\$\{/.test(read(f)))
    .map(rel).filter((f) => f !== 'utils/vastReferral.js')
  assert.deepEqual(offenders, [], 'a tagged link written by hand bypasses the one-place rule')
})

/* The two "create an account" moments of the product, and nothing else. */
const SIGNUP_SURFACES = ['components/settings/TrainingSection.jsx', 'pages/SetupPage.jsx']

test('only the two "create an account" surfaces build the sign-up link, once each', () => {
  const callers = SOURCES.filter((f) => read(f).includes('vastSignupUrl('))
    .map(rel).filter((f) => f !== 'utils/vastReferral.js').sort()
  assert.deepEqual(callers, SIGNUP_SURFACES)
  for (const f of SIGNUP_SURFACES) {
    const calls = read(resolve(SRC, f)).match(/vastSignupUrl\(/g)
    assert.equal(calls.length, 1, `${f}: one call site — the "create an account" link, nothing else`)
  }
})

test('every tagged sign-up link renders the shared disclosure beside it, and the wording lives once', () => {
  for (const f of SIGNUP_SURFACES) {
    assert.ok(read(resolve(SRC, f)).includes('<VastReferralDisclosure'),
      `${f}: a tagged link without its disclosure is the thing this rule exists to prevent`)
  }
  const wording = SOURCES.filter((f) => read(f).includes('is a referral link: open a vast.ai account through it')).map(rel)
  assert.deepEqual(wording, ['components/common/VastReferralDisclosure.jsx'],
    'the disclosure sentence is written in one component so the two surfaces cannot drift')
})

test('Billing, Keys and the instances console stay untagged: those users already have an account', () => {
  const tagged = SOURCES.filter((f) => /https:\/\/cloud\.vast\.ai\/[a-z-]+\/\?/.test(read(f))).map(rel)
  assert.deepEqual(tagged, [], 'a query string on a console sub-page earns nothing and reads as tracking')
})

test('README and the settings guide mirror the id: once each, with the disclosure and the untagged link', () => {
  const readme = read(README)
  const guide = read(GUIDE)
  if (!VAST_REFERRAL_ID) {
    assert.deepEqual(taggedIds(readme), [], 'README carries a ref_id the app does not have')
    for (const doc of DOCS) {
      assert.deepEqual(taggedIds(read(doc)), [], `${relative(REPO, doc)} carries a ref_id the app does not have`)
    }
    assert.doesNotMatch(readme, /Affiliate disclosure/, 'a disclosure with no link to disclose')
    return
  }
  assert.deepEqual(taggedIds(readme), [VAST_REFERRAL_ID], 'README: the tagged link, exactly once')
  assert.deepEqual(taggedIds(guide), [VAST_REFERRAL_ID], 'settings guide: the tagged link, exactly once')
  for (const doc of DOCS) {
    if (doc !== GUIDE) assert.deepEqual(taggedIds(read(doc)), [], `${relative(REPO, doc)} must not carry the tag`)
  }
  assert.match(readme, /\*\*Affiliate disclosure\.\*\*/, 'README: the disclosure block under the API-keys table')
  assert.match(readme, UNTAGGED, 'README: the untagged link is what turns a disclosure into a choice')
  assert.match(readme, /no upsell[\s\S]{0,600}referral link/,
    'README: "no upsell" must say in what sense it stays true next to a referral link')
  assert.match(guide, /disclos/i, 'settings guide: points at the disclosure')
})
