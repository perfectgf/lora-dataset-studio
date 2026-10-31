import assert from 'node:assert/strict'
import test from 'node:test'
import { createElement, renderToStaticMarkup } from './support/mountJsx.mjs'

const { RunErrorMessage } = await import('../src/components/runs/RunError.jsx')
const render = (error, dedicatedToken = false, tokenPresence = 'missing') => renderToStaticMarkup(
  createElement(RunErrorMessage, { error, dedicatedToken, tokenPresence }))

const gatedError = '401 Client Error. (Request ID: example-request)\n\n'
  + 'Cannot access gated repo for url https://huggingface.co/example/model/resolve/main/raw.safetensors.\n'
  + 'Access to model example/model is restricted. You must have access to it and be authenticated to access it. Please log in.'

test('the reported gated error keeps its detail and links straight to the token and model access', () => {
  const html = render(gatedError)
  assert.match(html, /href="#\/settings\/local-tools\?focus=HF_TOKEN"/)
  assert.match(html, /href="https:\/\/huggingface\.co\/example\/model"/)
  assert.match(html, /browser login is separate from LDS/)
  assert.match(html, /401 Client Error/)
  assert.match(html, /example-request/)
  assert.match(html, /Please log in/)
  assert.doesNotMatch(html, /HF_CLOUD_TOKEN/)
  assert.match(html, /href="https:\/\/huggingface\.co\/settings\/tokens\/new\?tokenType=read"/)
  assert.match(html, /type="password"/)
  assert.match(html, /Save token/)
})

test('full-model cloud failures open the dedicated token, not the general token', () => {
  const html = render(gatedError, true)
  assert.match(html, /href="#\/plugins\/cloud_training\/settings\?focus=HF_CLOUD_TOKEN"/)
  assert.doesNotMatch(html, /focus=HF_TOKEN/)
  assert.doesNotMatch(html, /tokenType=read|type="password"/)
})

test('a configured token hides setup on historical errors without claiming model access is fixed', () => {
  const html = render(gatedError, false, 'configured')
  assert.match(html, /A Hugging Face token is configured/)
  assert.match(html, /previous attempt/)
  assert.match(html, /Retry to check access/)
  assert.match(html, /<details><summary[^>]*>Recorded error from this attempt/)
  assert.match(html, /401 Client Error/)
  assert.doesNotMatch(html, /<details open|type="password"|Create a read token|Add or update|Save token/)
  assert.match(html, /focus=HF_TOKEN/)
})

test('unknown settings never masquerade as a missing or configured token', () => {
  for (const state of ['loading', 'unknown']) {
    const html = render(gatedError, false, state)
    assert.doesNotMatch(html, /type="password"|Create a read token|A Hugging Face token is configured/)
    assert.match(html, /focus=HF_TOKEN/)
  }
})

test('403 offers both token permissions and model access without claiming the licence is the only cause', () => {
  const html = render(gatedError.replace('401', '403'))
  assert.match(html, /token permissions/)
  assert.match(html, /focus=HF_TOKEN/)
  assert.match(html, /Check model access/)
})

test('gated failures without a status code still have a recovery link', () => {
  assert.match(render('GatedRepoError: Cannot access gated repo.'), /focus=HF_TOKEN/)
})

test('unrelated authentication failures and Hugging Face transfer errors get no token advice', () => {
  for (const error of [
    '401 Client Error for url https://example.org/api',
    '403 Forbidden: cloud provider key rejected',
    'Connection timed out for https://huggingface.co/example/model/resolve/main/raw.safetensors',
    'Hugging Face download failed: no space left on device',
  ]) {
    assert.doesNotMatch(render(error), /<a\b|Configure Hugging Face token/)
  }
})

test('missing tokens mentioned by name are actionable without a download URL', () => {
  const html = render('HF_TOKEN is missing. Please configure a Hugging Face token.')
  assert.match(html, /focus=HF_TOKEN/)
  assert.doesNotMatch(html, /Check model access/)
})

test('model links drop file paths and query credentials and never link to lookalike hosts', () => {
  const html = render(gatedError.replace('raw.safetensors.', 'raw.safetensors?token=example-secret'))
  const hrefs = [...html.matchAll(/href="([^"]+)"/g)].map(match => match[1])
  assert.deepEqual(hrefs, ['#/settings/local-tools?focus=HF_TOKEN', 'https://huggingface.co/example/model',
    'https://huggingface.co/settings/tokens/new?tokenType=read'])
  for (const host of ['huggingface.co.example.org', 'huggingface.co@evil.example']) {
    assert.doesNotMatch(render(gatedError.replace('huggingface.co', host)), /Check model access/)
  }
})
