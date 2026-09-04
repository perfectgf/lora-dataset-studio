import assert from 'node:assert/strict'
import { readFileSync, readdirSync, statSync } from 'node:fs'
import { dirname, join, relative, resolve } from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

import { VAST_CONSOLE_URL, VAST_REFERRAL_ID, vastSignupUrl } from './vastReferral.js'

/* The referral link is a material interest disclosed next to each link that
 * asks someone to CREATE a vast.ai account, with the untagged link beside it.
 * Everything below pins the shape that keeps that honest:
 *   · the id lives in vastReferral.js and nowhere else in the sources — in
 *     any spelling, not only the literal URL;
 *   · exactly two surfaces build the sign-up link, once each, and each renders
 *     the shared disclosure beside it; whether the disclosure shows is decided
 *     INSIDE the disclosure (empty id → nothing), never by the caller;
 *   · the escape hatches (Billing, Keys, instances console) stay untagged;
 *   · README and the settings guide carry the SAME id, once each, with the
 *     disclosure and the untagged link — or no id at all when none is set.
 * The rendered surfaces are covered by tests/vast-key-guide-render.test.mjs.
 * Known, accepted: the What's-new entry is static upstream history and keeps
 * describing the referral link in a fork that blanks the id. */

const HERE = dirname(fileURLToPath(import.meta.url))
const SRC = resolve(HERE, '..')             // frontend/src
const REPO = resolve(SRC, '../..')          // repo root
const read = (abs) => readFileSync(abs, 'utf8')

/* Global regex: used ONLY through matchAll (a `/g` regex shared across .test()
   calls carries its lastIndex from one file to the next and skips matches). */
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

const MODULE = 'utils/vastReferral.js'
const DISCLOSURE = 'components/common/VastReferralDisclosure.jsx'
/* The two "create an account" moments of the product, and nothing else. */
const SIGNUP_SURFACES = ['components/settings/TrainingSection.jsx', 'components/setup/CloudSignupNote.jsx']

test('no id → the untagged console; an id → exactly one ref_id on the console root', () => {
  assert.equal(vastSignupUrl(''), VAST_CONSOLE_URL)
  assert.equal(vastSignupUrl(null), VAST_CONSOLE_URL)
  assert.equal(vastSignupUrl('   '), VAST_CONSOLE_URL)
  assert.equal(vastSignupUrl(undefined), vastSignupUrl(VAST_REFERRAL_ID))
  assert.equal(vastSignupUrl(' 12345 '), 'https://cloud.vast.ai/?ref_id=12345')
  assert.equal(vastSignupUrl(12345), 'https://cloud.vast.ai/?ref_id=12345')
  assert.equal(vastSignupUrl('a b&c'), 'https://cloud.vast.ai/?ref_id=a%20b%26c')
})

test('the sources never spell a tagged vast.ai link out, in any form — vastSignupUrl is the only builder', () => {
  const offenders = SOURCES.filter((f) => {
    const s = read(f)
    return taggedIds(s).length > 0 || /ref_id=/.test(s) || /cloud\.vast\.ai\/\?/.test(s)
  }).map(rel).filter((f) => f !== MODULE)
  assert.deepEqual(offenders, [], 'a tagged link written by hand (template, concat, other scheme) bypasses the one-place rule')
})

test('the referral module is imported by the disclosure and the two sign-up surfaces, nowhere else', () => {
  const importers = SOURCES.filter((f) => /from '[^']*utils\/vastReferral(\.js)?'/.test(read(f))).map(rel).sort()
  assert.deepEqual(importers, [DISCLOSURE, ...SIGNUP_SURFACES].sort(),
    'a third importer is a third surface (or an alias around vastSignupUrl) — extend the list on purpose, never by accident')
})

test('each surface builds the sign-up link once and renders the shared disclosure beside it; the disclosure decides its own visibility', () => {
  for (const f of SIGNUP_SURFACES) {
    const s = read(resolve(SRC, f))
    assert.equal((s.match(/vastSignupUrl\(/g) || []).length, 1, `${f}: one call site — the "create an account" link, nothing else`)
    assert.equal((s.match(/<VastReferralDisclosure\b/g) || []).length, 1, `${f}: the disclosure, once, beside the link`)
    assert.ok(!s.includes('VAST_CONSOLE_URL'), `${f}: no hand-coded "is it tagged?" — VastReferralDisclosure decides, in one place`)
  }
  const disclosure = read(resolve(SRC, DISCLOSURE))
  assert.ok(disclosure.includes('return null'), 'the disclosure renders nothing for an untagged link')
  const wording = SOURCES.filter((f) => read(f).includes('is a referral link: open a vast.ai account through it')).map(rel)
  assert.deepEqual(wording, [DISCLOSURE], 'the disclosure sentence is written in one component so the two surfaces cannot drift')
})

test('the Setup wizard mounts the note component instead of building the link inline', () => {
  const setup = read(resolve(SRC, 'pages/SetupPage.jsx'))
  assert.equal((setup.match(/<CloudSignupNote\b/g) || []).length, 1, 'SetupPage renders the note once')
  assert.ok(!setup.includes('vastSignupUrl('), 'SetupPage never builds the link itself: the page cannot be rendered by a test, the component can')
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
  assert.match(guide, UNTAGGED, 'settings guide: the untagged link beside the tagged one')
  /* The guide is also rendered INSIDE the app (GuidePage, the help modal) by the
     in-house Markdown.jsx, which knows [text](url) but not <autolinks>, and which
     opens hrefs as-is: a relative README path lands on a 404 there. */
  assert.doesNotMatch(guide, /<https:\/\/cloud\.vast\.ai\/>/, 'settings guide: never an autolink — plain text in-app')
  assert.doesNotMatch(guide, /\]\(\.\.\/\.\.\/README\.md#getting-api-keys\)/, 'settings guide: no relative README link — a 404 in-app')
})
