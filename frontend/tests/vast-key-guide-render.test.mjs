/**
 * What the API-key guide actually RENDERS around the vast.ai referral link.
 *
 * The rule is a disclosure rule, and a disclosure is a property of the markup:
 * the tagged link sits on the one step that asks for an account, the text
 * that says so comes right after the steps (not in a tooltip, not on another
 * page), and the untagged link is one click away. The pure module cannot see
 * any of that — only a render can — and the same render proves that with no
 * id at all the guide is byte-for-byte the plain walkthrough.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { render } from './support/mountJsx.mjs'

const { VastKeyGuide } = await import('../src/components/settings/TrainingSection.jsx')

const hrefs = (html) => [...html.matchAll(/href="([^"]+)"/g)].map((m) => m[1])

test('no referral id: the plain walkthrough — untagged sign-up link, no disclosure', () => {
  const html = render(VastKeyGuide, { referralId: '' })
  assert.deepEqual(hrefs(html), [
    'https://cloud.vast.ai/',
    'https://cloud.vast.ai/billing/',
    'https://cloud.vast.ai/manage-keys/',
  ])
  assert.doesNotMatch(html, /referral/i, 'nothing to disclose, nothing disclosed')
})

test('with a referral id: step 1 carries it, the disclosure follows the steps, the untagged link stays beside it', () => {
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
  assert.match(html, /pays this project 3%/)
  assert.match(html, /costs you nothing/)
  assert.match(html, /works exactly the same/, 'the untagged link is offered as a real alternative')
})

test('the default render uses the id shipped in vastReferral.js', async () => {
  const { VAST_REFERRAL_ID, vastSignupUrl } = await import('../src/utils/vastReferral.js')
  const html = render(VastKeyGuide)
  assert.equal(hrefs(html)[0], vastSignupUrl(VAST_REFERRAL_ID))
  assert.equal(/referral link/.test(html), VAST_REFERRAL_ID !== '')
})
