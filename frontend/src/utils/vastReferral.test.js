import assert from 'node:assert/strict'
import { readFileSync, readdirSync, statSync } from 'node:fs'
import { dirname, join, relative, resolve } from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

import { VAST_CONSOLE_URL, VAST_REFERRAL_ID, vastSignupUrl, vastUrl } from './vastReferral.js'

/* The maintainer's rule (2026-09-05): wherever we talk about vast.ai, our link
 * goes with it. Everything below pins the shape that keeps that true AND honest:
 *   · the id lives in vastReferral.js and nowhere else in the sources — no
 *     vast.ai URL is spelled out anywhere, every link is built by vastUrl();
 *   · the files allowed to build or render such a link are listed here, so a
 *     new surface is a decision, not an accident;
 *   · the disclosure renders beside the two "create an account" moments,
 *     decides its own visibility (empty id → nothing) and offers the ONE
 *     deliberately untagged link; its wording is written once;
 *   · README, the guides and .env.example carry the SAME id on every vast.ai
 *     URL, and every section that mentions vast.ai carries at least one such
 *     link — or no id at all anywhere when none is set (forks).
 * The rendered surfaces are covered by tests/vast-key-guide-render.test.mjs.
 * Known, accepted: the What's-new entry is static upstream history and keeps
 * describing the referral links in a fork that blanks the id. */

const HERE = dirname(fileURLToPath(import.meta.url))
const SRC = resolve(HERE, '..')             // frontend/src
const REPO = resolve(SRC, '../..')          // repo root
const read = (abs) => readFileSync(abs, 'utf8')

/* Global regexes are used ONLY through matchAll (a `/g` regex shared across
   .test() calls carries its lastIndex from one file to the next). */
const TAGGED = /https:\/\/cloud\.vast\.ai\/[a-z/-]*\?ref_id=([^)\s"'<>&]+)/g
const ANY_VAST_URL = /https?:\/\/(?:[a-z0-9-]+\.)*vast\.ai\/?[^\s)"'<>`]*/gi
/* The bare console root: not followed by a query string or a sub-path. */
const UNTAGGED_ROOT = /https:\/\/cloud\.vast\.ai\/(?![?a-z])/

function walk(dir, keep, out = []) {
  for (const name of readdirSync(dir)) {
    if (name === 'node_modules' || name === 'dist' || name === 'superpowers') continue
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
const ENV_EXAMPLE = resolve(REPO, '.env.example')

const rel = (abs) => relative(SRC, abs).replace(/\\/g, '/')
const relRepo = (abs) => relative(REPO, abs).replace(/\\/g, '/')
const taggedIds = (text) => [...text.matchAll(TAGGED)].map((m) => m[1])
const vastUrls = (text) => [...text.matchAll(ANY_VAST_URL)].map((m) => m[0])

const MODULE = 'utils/vastReferral.js'
const DISCLOSURE = 'components/common/VastReferralDisclosure.jsx'
const LINK = 'components/common/VastLink.jsx'
/* The two "create an account" moments of the product. */
const SIGNUP_SURFACES = ['components/settings/TrainingSection.jsx', 'components/setup/CloudSignupNote.jsx']
/* Every file that may import the referral module (build a URL), and every file
   that may render <VastLink>. Extend on purpose, never by accident. */
const MODULE_IMPORTERS = [DISCLOSURE, LINK, ...SIGNUP_SURFACES, 'pages/CloudRunsPage.jsx'].sort()
const LINK_USERS = ['components/dataset/TrainingPanel.jsx', 'components/settings/TrainingSection.jsx', 'pages/CloudRunsPage.jsx'].sort()

test('vastUrl tags any console page with the id; no id → the plain page; the sign-up link is the root', () => {
  assert.equal(vastUrl('/', ''), VAST_CONSOLE_URL)
  assert.equal(vastUrl(undefined, null), VAST_CONSOLE_URL)
  assert.equal(vastUrl('/billing/', ''), 'https://cloud.vast.ai/billing/')
  assert.equal(vastUrl('/billing/', '12345'), 'https://cloud.vast.ai/billing/?ref_id=12345')
  assert.equal(vastUrl('instances/', ' 12345 '), 'https://cloud.vast.ai/instances/?ref_id=12345')
  assert.equal(vastUrl('/', 12345), 'https://cloud.vast.ai/?ref_id=12345')
  assert.equal(vastUrl('/', 'a b&c'), 'https://cloud.vast.ai/?ref_id=a%20b%26c')
  assert.equal(vastSignupUrl('12345'), vastUrl('/', '12345'))
  assert.equal(vastSignupUrl(), vastUrl('/', VAST_REFERRAL_ID))
})

test('the sources never spell a vast.ai URL or a ref_id out — vastUrl is the only builder', () => {
  const offenders = SOURCES.filter((f) => {
    const s = read(f)
    return vastUrls(s).length > 0 || /ref_id=/.test(s) || /cloud\.vast\.ai\//.test(s)
  }).map(rel).filter((f) => f !== MODULE)
  assert.deepEqual(offenders, [], 'a vast.ai URL written by hand (any spelling) bypasses the one-place rule')
})

test('the referral module and <VastLink> are used by the listed files, nowhere else', () => {
  const importers = SOURCES.filter((f) => /from '[^']*utils\/vastReferral(\.js)?'/.test(read(f))).map(rel).sort()
  assert.deepEqual(importers, MODULE_IMPORTERS, 'a new importer is a new surface — list it on purpose')
  const users = SOURCES.filter((f) => read(f).includes('<VastLink')).map(rel).sort()
  assert.deepEqual(users, LINK_USERS, 'a new <VastLink> user is a new surface — list it on purpose')
})

test('the disclosure decides its own visibility, renders once beside each sign-up link, and its wording lives once', () => {
  const disclosure = read(resolve(SRC, DISCLOSURE))
  assert.ok(disclosure.includes('return null'), 'the disclosure renders nothing for untagged links')
  for (const f of SIGNUP_SURFACES) {
    const s = read(resolve(SRC, f))
    assert.equal((s.match(/vastSignupUrl\(/g) || []).length, 1, `${f}: the sign-up link, built once`)
    assert.equal((s.match(/<VastReferralDisclosure\b/g) || []).length, 1, `${f}: the disclosure, once, beside the link`)
    assert.ok(!s.includes('VAST_CONSOLE_URL'), `${f}: no hand-coded "is it tagged?" — VastReferralDisclosure decides, in one place`)
  }
  const wording = SOURCES.filter((f) => read(f).includes('are referral links: open a vast.ai account through one of them')).map(rel)
  assert.deepEqual(wording, [DISCLOSURE], 'the disclosure sentence is written in one component so the surfaces cannot drift')
})

test('the Setup wizard mounts the note component instead of building the link inline', () => {
  const setup = read(resolve(SRC, 'pages/SetupPage.jsx'))
  assert.equal((setup.match(/<CloudSignupNote\b/g) || []).length, 1, 'SetupPage renders the note once')
  assert.ok(!setup.includes('vastSignupUrl('), 'SetupPage never builds the link itself: the page cannot be rendered by a test, the component can')
})

/* Markdown sections, with fenced code blocks left out. */
function sections(md) {
  const out = []
  let cur = { heading: '(top)', lines: [] }
  let inCode = false
  for (const line of md.split('\n')) {
    if (line.trim().startsWith('```')) { inCode = !inCode; continue }
    if (inCode) continue
    if (/^#{1,6}\s/.test(line)) { out.push(cur); cur = { heading: line.trim(), lines: [] }; continue }
    cur.lines.push(line)
  }
  out.push(cur)
  return out
}
/* A line with its code spans, links, URLs, autolinks and italic-quoted UI strings blanked. */
const masked = (line) => line
  .replace(/`[^`]*`/g, ' ').replace(/\[[^\]]*\]\([^)]*\)/g, ' ').replace(/https?:\/\/\S+/g, ' ')
  .replace(/\*"[^"]*"\*/g, ' ').replace(/<[^>]*>/g, ' ')
const mentionsVast = (lines) => lines.some((l) => /(?<![\w./-])vast\.ai(?![\w-])/i.test(masked(l)))

test('README, the guides and .env.example: every vast.ai URL carries the id, and every section that talks about vast.ai carries the link', () => {
  const textFiles = [README, ENV_EXAMPLE, ...DOCS]
  if (!VAST_REFERRAL_ID) {
    for (const f of textFiles) assert.ok(!/ref_id=/.test(read(f)), `${relRepo(f)} carries a ref_id the app does not have`)
    assert.doesNotMatch(read(README), /Affiliate disclosure/, 'a disclosure with no link to disclose')
    return
  }
  for (const f of textFiles) {
    const text = read(f)
    for (const line of text.split('\n')) {
      if (/untagged/i.test(line)) continue                       // the disclosure's escape hatch
      for (const url of vastUrls(line)) {
        if (/console\.vast\.ai\/api/.test(url)) continue         // API endpoints, not links
        assert.ok(url.includes(`ref_id=${VAST_REFERRAL_ID}`), `${relRepo(f)}: untagged vast.ai link ${url}`)
      }
    }
    for (const id of taggedIds(text)) assert.equal(id, VAST_REFERRAL_ID, `${relRepo(f)}: a foreign referral id`)
  }
  for (const f of [README, ...DOCS]) {
    for (const s of sections(read(f))) {
      if (!mentionsVast(s.lines)) continue
      assert.ok(s.lines.some((l) => l.includes(`ref_id=${VAST_REFERRAL_ID}`)),
        `${relRepo(f)} — section "${s.heading}" talks about vast.ai without our link`)
    }
  }
  const readme = read(README)
  assert.match(readme, /\*\*Affiliate disclosure\.\*\*/, 'README: the disclosure block under the API-keys table')
  assert.match(readme, UNTAGGED_ROOT, 'README: the untagged link is what turns a disclosure into a choice')
  assert.match(readme, /no upsell[\s\S]{0,600}referral links/, 'README: "no upsell" must say in what sense it stays true')
  const guide = read(GUIDE)
  assert.match(guide, /referral links/, 'settings guide: says its vast.ai links are referral links')
  assert.match(guide, UNTAGGED_ROOT, 'settings guide: the untagged link beside the tagged ones')
  /* The guide is also rendered INSIDE the app (GuidePage, the help modal) by the
     in-house Markdown.jsx, which knows [text](url) but not <autolinks>, and which
     opens hrefs as-is: a relative README path lands on a 404 there. */
  assert.doesNotMatch(guide, /<https:\/\/cloud\.vast\.ai\/>/, 'settings guide: never an autolink — plain text in-app')
  assert.doesNotMatch(guide, /\]\(\.\.\/\.\.\/README\.md/, 'settings guide: no relative README link — a 404 in-app')
})
