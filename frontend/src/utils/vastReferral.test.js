import assert from 'node:assert/strict'
import { readFileSync, readdirSync, statSync } from 'node:fs'
import { dirname, join, relative, resolve } from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

import { GUIDE as CLOUD_GUIDE } from '../../../bundled/cloud_training/frontend/guide.js'

import { VAST_CONSOLE_URL, VAST_REFERRAL_ID, vastSignupUrl, vastUrl } from './vastReferral.js'

/* The maintainer's rule (2026-09-05): wherever we talk about vast.ai, our link
 * goes with it. Everything below pins the shape that keeps that true AND honest:
 *   · the id lives in vastReferral.js and nowhere else in the sources — no
 *     vast.ai URL is spelled out anywhere, every link is built by vastUrl();
 *   · the files allowed to build or render such a link are listed here, so a
 *     new surface is a decision, not an accident;
 *   · the disclosure renders beside the two "create an account" moments,
 *     decides its own visibility (empty id → nothing); its wording is written once;
 *   · README, the guides and .env.example carry the SAME id on every vast.ai
 *     URL, and every section that mentions vast.ai carries at least one such
 *     link — or no id at all anywhere when none is set (forks).
 * The rendered surfaces are covered by tests/vast-key-guide-render.test.mjs.
 * Known, accepted: the What's-new entry is static upstream history and keeps
 * describing the referral links in a fork that blanks the id. */

const HERE = dirname(fileURLToPath(import.meta.url))
const SRC = resolve(HERE, '..')             // frontend/src
const REPO = resolve(SRC, '../..')          // repo root
const read = (abs) => readFileSync(abs, 'utf8').replace(/\r\n/g, '\n')

/* Global regexes are used ONLY through matchAll (a `/g` regex shared across
   .test() calls carries its lastIndex from one file to the next). */
const TAGGED = /https:\/\/cloud\.vast\.ai\/[a-z/-]*\?ref_id=([^)\s"'<>&]+)/g
const ANY_VAST_URL = /https?:\/\/(?:[a-z0-9-]+\.)*vast\.ai\/?[^\s)"'<>`]*/gi

function walk(dir, keep, out = []) {
  for (const name of readdirSync(dir)) {
    if (name === 'node_modules' || name === 'dist' || name === 'superpowers') continue
    const abs = join(dir, name)
    if (statSync(abs).isDirectory()) walk(abs, keep, out)
    else if (keep(abs)) out.push(abs)
  }
  return out
}

const isCode = f => /\.(jsx?|mjs)$/.test(f) && !/\.test\.(jsx?|mjs)$/.test(f)
const SOURCES = [...walk(SRC, isCode),
  ...walk(resolve(REPO, 'bundled'), f => isCode(f) && /[/\\]frontend[/\\]/.test(f)
    && !/[/\\](guide|whatsNew|migratedNews)\.js$/.test(f))]

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
const SIGNUP_SURFACES = ['../../bundled/cloud_training/frontend/settings/CloudTrainingGroup.jsx',
  '../../bundled/cloud_training/frontend/setup/CloudSignupNote.jsx']
/* Every file that may import the referral module (build a URL), and every file
   that may render <VastLink>. Extend on purpose, never by accident. */
const MODULE_IMPORTERS = [DISCLOSURE, LINK, 'plugins/runtimeHost.jsx', 'plugins/guideContent.js',
  ...SIGNUP_SURFACES, '../../bundled/cloud_training/frontend/CloudRunsHub.jsx'].sort()
const LINK_USERS = ['../../bundled/cloud_training/frontend/settings/CloudTrainingGroup.jsx',
  '../../bundled/cloud_training/frontend/CloudRunsHub.jsx'].sort()

test('vastUrl tags any console page with the id; no id → the plain page; the sign-up link is the root', () => {
  assert.equal(vastUrl('/', ''), VAST_CONSOLE_URL)
  assert.equal(vastUrl(undefined, null), VAST_CONSOLE_URL)
  assert.equal(vastUrl('/billing/', ''), 'https://cloud.vast.ai/billing/')
  assert.equal(vastUrl('/billing/', '12345'), 'https://cloud.vast.ai/billing/?ref_id=12345')
  assert.equal(vastUrl('instances/', ' 12345 '), 'https://cloud.vast.ai/instances/?ref_id=12345')
  assert.equal(vastUrl('/', 12345), 'https://cloud.vast.ai/?ref_id=12345')
  assert.equal(new URL(vastUrl('/', 'a b&c')).searchParams.get('ref_id'), 'a b&c')
  assert.equal(vastSignupUrl('12345'), vastUrl('/', '12345'))
  assert.equal(vastSignupUrl(), vastUrl('/', VAST_REFERRAL_ID))
})

test('existing queries and fragments keep their destination and receive exactly the configured referral', () => {
  const url = new URL(vastUrl('/instances/?status=running&ref_id=other&ref_id=duplicate#offers'))
  assert.equal(url.pathname, '/instances/')
  assert.equal(url.searchParams.get('status'), 'running')
  assert.equal(url.hash, '#offers')
  assert.deepEqual(url.searchParams.getAll('ref_id'), [VAST_REFERRAL_ID])
  const plain = new URL(vastUrl('/billing/?ref_id=old&tab=credit#balance', ''))
  assert.equal(plain.searchParams.has('ref_id'), false)
  assert.equal(plain.searchParams.get('tab'), 'credit')
  assert.equal(plain.hash, '#balance')
})

test('the sources never spell a vast.ai URL or a ref_id out — vastUrl is the only builder', () => {
  const offenders = SOURCES.filter((f) => {
    const s = read(f)
    return vastUrls(s).length > 0 || /ref_id=/.test(s) || /cloud\.vast\.ai\//.test(s)
  }).map(rel).filter((f) => f !== MODULE)
  assert.deepEqual(offenders, [], 'a vast.ai URL written by hand (any spelling) bypasses the one-place rule')
})

test('the referral module and <VastLink> are used by the listed files, nowhere else', () => {
  const importers = SOURCES.filter((f) => /from ['"][^'"]*(?:utils\/vastReferral(?:\.js)?|@lds\/plugin-sdk\/links)['"]/.test(read(f))).map(rel).sort()
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

test('the Cloud preparation card is mounted by its product settings, through the declared slot', () => {
  const host = read(resolve(SRC, 'pages/pluginSettingsGroups.jsx'))
  const contributions = read(resolve(REPO, 'bundled/cloud_training/frontend/settings/contributions.js'))
  assert.match(contributions, /'setup.card':[\s\S]*?import\('\.\.\/setup\/CloudSignupNote\.jsx'\)/)
  assert.match(host, /contributions\('setup.card', 'setup'\).filter\(item => item.plugin === pluginId\)/)
  assert.match(host, /importer=\{card.panel\}/)
  assert.doesNotMatch(read(resolve(SRC, 'pages/SetupPage.jsx')), /<CloudSignupNote|vastSignupUrl\(/)
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

test('README, the core and owned guides and .env.example keep the same disclosed referral links', () => {
  // Moved guide sections remain in the census: the core chapter alone can no
  // longer prove the product's links, and generated JS is not executable UI.
  const textFiles = [README, ENV_EXAMPLE, ...DOCS].map(f => ({ name: relRepo(f), text: read(f) }))
  textFiles.push(...walk(resolve(REPO, 'bundled'), f => /[/\\]frontend[/\\]guide\.js$/.test(f))
    .map(f => ({ name: relRepo(f), text: read(f) })))
  const productDocs = CLOUD_GUIDE.sections.map(section => ({
    name: `bundled/cloud_training/frontend/guide.js#${section.chapter}/${section.anchor}`, text: section.markdown,
  }))
  textFiles.push(...productDocs)
  if (!VAST_REFERRAL_ID) {
    for (const { name, text } of textFiles) assert.ok(!/ref_id=/.test(text), `${name} carries an id the app does not have`)
    assert.doesNotMatch(read(README), /Affiliate disclosure/)
    return
  }
  for (const { name, text } of textFiles) {
    for (const line of text.split('\n')) {
      for (const url of vastUrls(line)) {
        if (/console\.vast\.ai\/api/.test(url)) continue
        assert.equal(new URL(url).searchParams.get('ref_id'), VAST_REFERRAL_ID, `${name}: wrong or missing referral on ${url}`)
      }
    }
    for (const id of taggedIds(text)) assert.equal(id, VAST_REFERRAL_ID, `${name}: a foreign referral id`)
  }
  for (const { name, text } of [...[README, ...DOCS].map(f => ({ name: relRepo(f), text: read(f) })), ...productDocs]) {
    for (const section of sections(text)) {
      if (!mentionsVast(section.lines)) continue
      assert.ok(section.lines.some(line => line.includes(`ref_id=${VAST_REFERRAL_ID}`)),
        `${name} — section "${section.heading}" talks about vast.ai without the link`)
    }
  }
  const readme = read(README)
  const disclosure = readme.match(/\*\*Affiliate disclosure[.:]\*\*[^\n]+/)?.[0]
  assert.ok(disclosure, 'README keeps its visible affiliate disclosure')
  assert.match(disclosure, /3% of referred users' spending/)
  assert.match(disclosure, /lifetime of their account/)
  assert.match(disclosure, /at no extra cost/)
  assert.match(disclosure, /billed directly by vast\.ai/)
  assert.ok(disclosure.includes(`ref_id=${VAST_REFERRAL_ID}`))
  const guide = read(GUIDE) + CLOUD_GUIDE.sections
    .filter(section => section.chapter === 'settings-reference')
    .map(section => section.markdown).join('\n')
  assert.match(guide, /referral links/)
  assert.doesNotMatch(guide, /<https:\/\/cloud\.vast\.ai\/>/)
  assert.doesNotMatch(guide, /\]\(\.\.\/\.\.\/README\.md/)
})
