import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

import {
  failureView, GENERIC_CAUSES, NO_ERROR_NOTE, FULL_LOG_NOTE, EMPTY_LOG_NOTE,
} from '../src/components/dataset/trainingFailure.js'

// A crash whose log holds nothing error-like: the exact shape the backend now
// sends for the huggingface_hub FutureWarning case (wannadecryptor, Discord).
const noise = {
  rc: 1,
  log_tail: 'constants.py:298: FutureWarning: HF_HUB_ENABLE_HF_TRANSFER is deprecated',
  excerpt: {
    kind: 'none', headline: '',
    text: 'constants.py:298: FutureWarning: HF_HUB_ENABLE_HF_TRANSFER is deprecated\nCaching latents to disk:   0%',
  },
}
const traceback = {
  rc: 1,
  excerpt: {
    kind: 'traceback',
    headline: 'RuntimeError: CUDA error: no kernel image is available for execution on the device',
    text: 'Traceback (most recent call last):\nRuntimeError: CUDA error: no kernel image is available',
  },
}

test('a log with no error line is shown as context, never painted as the cause', () => {
  const view = failureView(noise)
  assert.equal(view.tone, 'neutral')
  assert.equal(view.note, NO_ERROR_NOTE)
  assert.match(view.note, /No error line in this log/)
  assert.match(view.note, /training\.log/)   // the full log stays reachable
})

test('both notes name a button rendered in the failure block itself', () => {
  // Previously they named the "📂 Run folder" button buried in the collapsed
  // "📦 Checkpoints & trained LoRAs" disclosure — a beginner went hunting, and
  // that button opened the checkpoints folder, NOT the one holding
  // training.log (wannadecryptor, Discord). The note must point at the button
  // sitting right next to it.
  for (const note of [NO_ERROR_NOTE, FULL_LOG_NOTE]) {
    assert.match(note, /Open run folder/)
  }
  // node:test runs cases after the module has evaluated, so `block` (read at the
  // bottom of this file) is available here.
  assert.ok(block.includes('/>Open run folder'),
    'the button the notes point at must live in the failure block')
})

test('the failure block opens the folder of the run that DIED, not the browsed one', () => {
  // No trainingRunSelection here on purpose: the persisted family/base/variant
  // are the crashed run's, whereas the checkpoint browser can be showing any
  // other run.
  const call = block.slice(block.indexOf('/>Open run folder') - 900,
                           block.indexOf('📂 Open run folder'))
  assert.match(call, /train\/open-folder/)
  assert.match(call, /\{ target: 'run' \}/)
  assert.doesNotMatch(call, /trainingRunSelection/)
  // 400 px: the button and its note share a row that must wrap, never squeeze.
  assert.match(call, /flex flex-wrap[^"]*gap/)
})

test('a traceback IS the cause and is styled as one', () => {
  const view = failureView(traceback)
  assert.equal(view.tone, 'error')
  assert.equal(view.note, FULL_LOG_NOTE)
  assert.match(view.excerpt, /no kernel image/)
})

test('a plain error line is a cause too', () => {
  const view = failureView({ rc: 1, excerpt: { kind: 'error', headline: 'x', text: 'OSError: x' } })
  assert.equal(view.tone, 'error')
})

test('the title names the exit code, and survives its absence', () => {
  assert.match(failureView(traceback).title, /ai-toolkit exited 1/)
  assert.match(failureView({ excerpt: traceback.excerpt }).title, /nothing is training now/)
  assert.doesNotMatch(failureView({ excerpt: traceback.excerpt }).title, /exited/)
})

test('a legacy payload (no excerpt) falls back to the tail but stays neutral', () => {
  // training_error states live an hour: a crash recorded before this shipped
  // must not re-assert the old "the last lines are the cause" lie.
  const view = failureView({ rc: 1, log_tail: 'some tail' })
  assert.equal(view.excerpt, 'some tail')
  assert.equal(view.tone, 'neutral')
})

test('the GPU-architecture verdict replaces the generic guesswork', () => {
  const view = failureView({
    ...traceback,
    gpu_arch: { message: 'RTX 5070 is compute capability 12.0 (sm_120)…', command: 'python -m pip install …' },
  })
  assert.ok(view.gpuArch)
  assert.equal(view.causes, '', 'a proven cause must not be buried under "common first-run causes"')
})

test('without a GPU verdict the generic causes are still offered', () => {
  assert.equal(failureView(noise).causes, GENERIC_CAUSES)
  assert.match(GENERIC_CAUSES, /Hugging Face token/)
})

test('an empty gpu_arch payload is ignored rather than rendered blank', () => {
  assert.equal(failureView({ ...traceback, gpu_arch: {} }).gpuArch, null)
  assert.equal(failureView({ ...traceback, gpu_arch: { message: '' } }).gpuArch, null)
})

test('a run that logged nothing does not pretend to quote its last lines', () => {
  const view = failureView({ rc: 1, excerpt: { kind: 'none', headline: '', text: '' } })
  assert.equal(view.excerpt, '')
  assert.equal(view.note, EMPTY_LOG_NOTE)
  assert.doesNotMatch(view.note, /last lines/)
  assert.match(view.note, /Settings ▸ Local tools/)
})

test('no error at all renders nothing', () => {
  assert.equal(failureView(null), null)
  assert.equal(failureView(undefined), null)
})

// --- rendering contract: it has to survive a 400 px phone ------------------

const panel = readFileSync(new URL('../src/components/dataset/TrainingPanel.jsx', import.meta.url), 'utf8')
const block = panel.slice(panel.indexOf('const view = failureView(status.error)'),
                          panel.indexOf('{/* Live progress of THIS'))

test('the failure block is wired to the helper, not to the raw tail', () => {
  assert.ok(block.length > 200, 'failure block not found in TrainingPanel')
  assert.doesNotMatch(block, /status\.error\.log_tail/,
    'the panel must render view.excerpt, never the raw tail again')
  assert.match(block, /view\.tone === 'error'/)
})

test('every long string in the failure block wraps instead of scrolling the page', () => {
  for (const pre of block.match(/<pre className=\{?[`"][^`"]+[`"]/g) || []) {
    assert.match(pre, /whitespace-pre-wrap/, `a <pre> without wrapping: ${pre}`)
    assert.match(pre, /break-words|break-all/, `a <pre> that can overflow at 400px: ${pre}`)
  }
})

test('the remedy command is copyable without a horizontal page scroll', () => {
  const cmd = block.slice(block.indexOf('view.gpuArch.command &&'))
  assert.match(cmd, /overflow-x-auto/)
  assert.match(cmd, /break-all/)   // a pip index URL has no spaces to break on
})
