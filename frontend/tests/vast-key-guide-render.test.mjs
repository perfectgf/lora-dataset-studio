/**
 * What the two "create a vast.ai account" surfaces actually RENDER around the
 * referral link — and what the in-app Guide renders of the settings line.
 *
 * The rule is a disclosure rule, and a disclosure is a property of the markup:
 * the tagged link sits on the one step that asks for an account, the text
 * that says so comes right after it (not in a tooltip, not on another page),
 * and the untagged link is one click away. The pure module cannot see any of
 * that — only a render can — and the same renders prove that with no id at
 * all each surface is byte-for-byte the plain walkthrough.
 */
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

import { render } from './support/mountJsx.mjs'

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '../..')

const { VAST_REFERRAL_ID, vastSignupUrl } = await import('../src/utils/vastReferral.js')
const { VastKeyGuide } = await import('../src/components/settings/TrainingSection.jsx')
const { default: CloudSignupNote } = await import('../src/components/setup/CloudSignupNote.jsx')
const { default: VastReferralDisclosure } = await import('../src/components/common/VastReferralDisclosure.jsx')
const { default: Markdown } = await import('../src/components/common/Markdown.jsx')

const hrefs = (html) => [...html.matchAll(/href="([^"]+)"/g)].map((m) => m[1])

test('the disclosure renders nothing for an untagged link, and a paragraph with the untagged link for a tagged one', () => {
  const none = render(VastReferralDisclosure, { referralId: '' })
  assert.deepEqual(hrefs(none), [])
  assert.doesNotMatch(none, /referral link/)
  const some = render(VastReferralDisclosure, { referralId: '12345', subject: 'That link', className: 'x' })
  assert.deepEqual(hrefs(some), ['https://cloud.vast.ai/'])
  assert.match(some, /<p class="x">That link is a referral link/)
  assert.match(some, /pays this\s+project 3%/)
  assert.match(some, /costs you nothing/)
  assert.match(some, /The untagged cloud\.vast\.ai link<\/a>\s*works exactly the same/, 'the untagged link is offered under its own name')
})

test('Settings guide, no referral id: the plain walkthrough — untagged sign-up link, no disclosure', () => {
  const html = render(VastKeyGuide, { referralId: '' })
  assert.deepEqual(hrefs(html), [
    'https://cloud.vast.ai/',
    'https://cloud.vast.ai/billing/',
    'https://cloud.vast.ai/manage-keys/',
  ])
  assert.doesNotMatch(html, /referral/i, 'nothing to disclose, nothing disclosed')
})

test('Settings guide, with a referral id: step 1 carries it, the disclosure follows the steps, the untagged link stays beside it', () => {
  const html = render(VastKeyGuide, { referralId: '12345' })
  assert.deepEqual(hrefs(html), [
    'https://cloud.vast.ai/?ref_id=12345',
    'https://cloud.vast.ai/billing/',
    'https://cloud.vast.ai/manage-keys/',
    'https://cloud.vast.ai/',
  ], 'only the "create an account" link is tagged; Billing and Keys stay bare')
  const steps = html.indexOf('</ol>')
  const disclosure = html.indexOf('referral link')
  assert.ok(steps > 0 && disclosure > steps, 'the disclosure sits right after the steps')
  assert.match(html, /works exactly the same/, 'the untagged link is offered as a real alternative')
})

test('Setup note, no referral id: one untagged link, no disclosure', () => {
  const html = render(CloudSignupNote, { referralId: '' })
  assert.deepEqual(hrefs(html), ['https://cloud.vast.ai/'])
  assert.doesNotMatch(html, /referral/i)
  assert.match(html, /No GPU\? You can skip this step/)
})

test('Setup note, with a referral id: the tagged link, then the disclosure with the untagged link', () => {
  const html = render(CloudSignupNote, { referralId: '12345' })
  assert.deepEqual(hrefs(html), ['https://cloud.vast.ai/?ref_id=12345', 'https://cloud.vast.ai/'])
  const note = html.indexOf('No GPU?')
  const disclosure = html.indexOf('That sign-up link is a referral link')
  assert.ok(note >= 0 && disclosure > note, 'the disclosure follows the note it discloses')
})

test('the default renders use the id shipped in vastReferral.js', () => {
  for (const Surface of [VastKeyGuide, CloudSignupNote]) {
    const html = render(Surface)
    assert.equal(hrefs(html)[0], vastSignupUrl(VAST_REFERRAL_ID))
    assert.equal(/referral link/.test(html), VAST_REFERRAL_ID !== '')
  }
})

test('the settings-guide line renders IN-APP with both links clickable and nothing relative', { skip: !VAST_REFERRAL_ID && 'no id shipped' }, () => {
  const guide = readFileSync(resolve(REPO, 'docs/guide/settings-reference.md'), 'utf8')
  const line = guide.split('\n').find((l) => l.includes('ref_id='))
  assert.ok(line, 'the guide names the tagged link')
  const html = render(Markdown, { source: line })
  const links = hrefs(html)
  assert.ok(links.includes(vastSignupUrl(VAST_REFERRAL_ID)), 'the tagged link is a real link in-app')
  assert.ok(links.includes('https://cloud.vast.ai/'), 'the untagged alternative is a real link in-app, not an <autolink> left as text')
  assert.ok(links.every((h) => /^https?:\/\//.test(h)), `every href is absolute — the SPA serves neither README.md nor docs/: ${links.join(' ')}`)
  assert.doesNotMatch(html, /&lt;https/, 'no autolink escaped into visible text')
})
