/**
 * ☁ The video lane's cloud launch owes the image lane's manners — pinned at the
 * SOURCE, because the sequence they describe is what stands between a click and
 * a rented pod, and `renderToStaticMarkup` runs none of it.
 *
 * Three rules, each one a bug the image lane already paid for:
 *  · every pod-renting POST goes through postWithConfirmations, so the server's
 *    confirmable `PARALLEL_RUN:` refusal reaches the user as a question — posted
 *    bare it was a dead error (cloud-launch-confirmation.test.mjs);
 *  · the launch window opens AFTER the cloud-lane preflight and its gate, never
 *    before — a blocker must stop the click before the offers are even fetched;
 *  · the GPU class the user chose rides as `gpu_name`, or the server picks the
 *    cheapest offer above the floors on its own, silently, which is the exact
 *    "chosen for the user on a rented-by-the-minute decision" the window ended.
 */
import test from 'node:test'
import assert from 'node:assert/strict'
import { readSource } from './support/readSource.mjs'

// Comments are stripped first: the block's own comment says what the window
// "replaced an inline <select>", and a source guard that reads prose finds the
// thing it forbids in the sentence explaining why it is gone. Code only.
const codeOnly = (text) => text
  .replace(/\{\/\*[\s\S]*?\*\/\}/g, '')
  .replace(/\/\*[\s\S]*?\*\//g, '')
  .replace(/^\s*\/\/.*$/gm, '')
const block = codeOnly(readSource('src/components/videobank/VideoTrainingBlock.jsx'))
const dialog = codeOnly(readSource('src/components/videobank/VideoCloudLaunchDialog.jsx'))

test('every pod-renting POST of the video block relays the confirmable refusals', () => {
  assert.match(block, /postWithConfirmations\(\(b\) => postJson\(url, b\), body, 'Launch anyway \(force\)'\)/,
    'postCloud must post through postWithConfirmations')
  // No bare postJson may rent a pod. The only postJson calls left are the local
  // lane's (train / stop), which rent nothing.
  const bare = [...block.matchAll(/await postJson\(([^,]+),/g)].map((m) => m[1].trim())
  for (const url of bare) {
    assert.ok(/\/train(`|'|\/stop)/.test(url) && !/cloud/i.test(url),
      `a bare postJson rents a pod without the confirmation loop: ${url}`)
  }
})

test('the launch window opens only after the cloud-lane preflight and its gate', () => {
  const open = block.slice(block.indexOf('const openCloudDialog = async () => {'),
    block.indexOf('const launchCloud ='))
  assert.ok(open.length > 0, 'openCloudDialog not found')
  const preflightAt = open.indexOf("videoPreflightUrl(ds.id, 'cloud')")
  const gateAt = open.indexOf("preflightGate(report, { lane: 'cloud' })")
  const blockersAt = open.indexOf('if (!gate.ok)')
  const confirmAt = open.indexOf('gate.confirmText && !window.confirm(gate.confirmText)')
  const openAt = open.indexOf('setCloudDialog(true)')
  for (const [name, at] of [['preflight', preflightAt], ['gate', gateAt],
    ['blockers', blockersAt], ['confirm', confirmAt], ['open', openAt]]) {
    assert.notEqual(at, -1, `${name} step missing from openCloudDialog`)
  }
  assert.ok(preflightAt < gateAt && gateAt < blockersAt && blockersAt < confirmAt && confirmAt < openAt,
    'the order must be: preflight → gate → blockers stop → warnings confirm → open')
  // The licence question comes FIRST — before anything is fetched or spent.
  assert.ok(open.indexOf('ensureLicenceAck(ds') < preflightAt, 'the licence ack must precede the preflight')
})

test('the chosen GPU class rides on the launch body, and the dialog is what chooses it', () => {
  assert.match(block, /const launchCloud = \(gpuName\) => postCloud\(videoDatasetCloudUrl\(ds\.id\),/)
  assert.match(block, /\.\.\.\(gpuName \? \{ gpu_name: gpuName \} : \{\}\)/)
  assert.match(block, /<VideoCloudLaunchDialog ds=\{ds\} steps=\{steps\} cloudStatus=\{cloudStatus\}/)
  assert.match(dialog, /const launched = await onLaunch\(selected\)/)
  assert.match(dialog, /if \(launched\) onClose\(\)/, 'the window closes only on a real success')
  // The inline picker that rented on a click is gone for good.
  assert.ok(!/Choose a GPU/.test(block), 'the inline GPU select must not come back')
  assert.ok(!/<select/.test(block), 'no select in the block — the dialog owns the choice')
})
